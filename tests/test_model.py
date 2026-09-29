"""lowbit.model on tiny random models (no downloads)."""
import json

import numpy as np
import pytest

from lowbit import model as M


@pytest.fixture(scope="module")
def tiny():
    cfg = M.tiny_config()
    return cfg, M.random_weights(cfg, seed=3)


def naive_forward(cfg, W, tokens):
    """Straightforward per-head reference (float64, explicit loops, no KV cache)."""
    g = lambda n: np.asarray(W[n], np.float64)  # noqa: E731
    T, D, H, KV = len(tokens), cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    x = g("model.embed_tokens.weight")[tokens]

    def norm(v, w):
        return v / np.sqrt((v * v).mean(-1, keepdims=True) + cfg.rms_norm_eps) * w

    def rope(v, t):  # v [D]: element i pairs with i + D/2, angle t * theta^(-2i/D)
        out = v.copy()
        for i in range(D // 2):
            a = t * cfg.rope_theta ** (-2 * i / D)
            out[i] = v[i] * np.cos(a) - v[i + D // 2] * np.sin(a)
            out[i + D // 2] = v[i + D // 2] * np.cos(a) + v[i] * np.sin(a)
        return out

    for li in range(cfg.num_layers):
        p = f"model.layers.{li}."
        h = norm(x, g(p + "input_layernorm.weight"))
        proj = {}
        for nm in "qkv":
            proj[nm] = h @ g(p + f"self_attn.{nm}_proj.weight").T
            if cfg.qkv_bias:
                proj[nm] = proj[nm] + g(p + f"self_attn.{nm}_proj.bias")
        att = np.zeros((T, H * D))
        for hh in range(H):
            kvh = hh // (H // KV)
            q = np.stack([rope(proj["q"][t, hh * D:(hh + 1) * D], t) for t in range(T)])
            k = np.stack([rope(proj["k"][t, kvh * D:(kvh + 1) * D], t) for t in range(T)])
            v = proj["v"][:, kvh * D:(kvh + 1) * D]
            for t in range(T):
                s = q[t] @ k[: t + 1].T / np.sqrt(D)
                pr = np.exp(s - s.max())
                att[t, hh * D:(hh + 1) * D] = (pr / pr.sum()) @ v[: t + 1]
        x = x + att @ g(p + "self_attn.o_proj.weight").T
        h = norm(x, g(p + "post_attention_layernorm.weight"))
        gt = h @ g(p + "mlp.gate_proj.weight").T
        up = h @ g(p + "mlp.up_proj.weight").T
        x = x + (gt / (1 + np.exp(-gt)) * up) @ g(p + "mlp.down_proj.weight").T
    x = norm(x, g("model.norm.weight"))
    head = g("model.embed_tokens.weight") if cfg.tie_word_embeddings else g("lm_head.weight")
    return x @ head.T


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_forward_shapes(tiny):
    cfg, W = tiny
    m = M.Model(cfg, W)
    toks = [1, 5, 9, 2]
    assert m.forward(toks, logits="all").shape == (4, cfg.vocab_size)
    assert m.forward(toks).shape == (cfg.vocab_size,)
    assert m.forward(toks, logits="none").shape == (4, cfg.hidden_size)
    c = m.new_cache(8)
    m.forward(toks, c)
    assert c.length == 4
    with pytest.raises(ValueError):
        m.forward(list(range(5)), c)  # 4 + 5 > 8


@pytest.mark.parametrize("over", [{}, dict(num_kv_heads=1, tie_word_embeddings=False,
                                          qkv_bias=False, head_dim=32, num_heads=8),
                                  dict(num_kv_heads=4)])
def test_matches_naive_reference(over):
    cfg = M.tiny_config(**over)
    W = M.random_weights(cfg, seed=7)
    toks = [3, 1, 4, 1, 5, 9, 2, 6]
    ref = naive_forward(cfg, W, toks)
    out = M.Model(cfg, W).forward(toks, logits="all")
    assert _rel(out, ref) < 1e-5


def test_kv_cache_decode_equals_full_forward(tiny):
    cfg, W = tiny
    m = M.Model(cfg, W)
    toks = (np.arange(40) * 7) % cfg.vocab_size
    full = m.forward(toks, logits="all")
    c = m.new_cache(64)
    parts = [m.forward(toks[:10], c, logits="all"), m.forward(toks[10:17], c, logits="all")]
    parts += [m.forward([t], c)[None] for t in toks[17:]]
    assert _rel(np.concatenate(parts), full) < 1e-5
    # greedy generation uses the same cache path
    g = m.generate(toks[:5], 6)
    manual, seq = [], list(toks[:5])
    for _ in range(6):
        manual.append(int(np.argmax(m.forward(seq, logits="last"))))
        seq.append(manual[-1])
    assert g == manual


@pytest.mark.parametrize("backend", ["w4a16", "w4a8", "mxfp4"])
def test_quantized_backend_close_to_fp32(tiny, backend):
    cfg, W = tiny
    toks = (np.arange(48) * 5) % cfg.vocab_size
    ref = M.Model(cfg, W).forward(toks, logits="all")
    q = M.Model(cfg, W, backend=backend)
    out = q.forward(toks, logits="all")
    assert _rel(out, ref) < 0.1
    assert np.mean(out.argmax(1) == ref.argmax(1)) >= 0.9
    assert q.linear_weight_bytes < M.Model(cfg, W).linear_weight_bytes / 5
    # every linear is exactly the library GEMM on its quantized weights
    lin = q.layers[0].gate_up
    x = np.random.default_rng(0).standard_normal((3, cfg.hidden_size)).astype(np.float32)
    deq = lin.wq.dequantize().astype(np.float64)
    tol = 0.05 if backend == "w4a8" else 1e-4
    assert _rel(lin(x), x.astype(np.float64) @ deq.T) < tol


@pytest.mark.parametrize("kw", [dict(asym=True), dict(int4_scale="signed-max"),
                                dict(quantize_lm_head=False)])
def test_int4_quantizer_variants(tiny, kw):
    cfg, W = tiny
    toks = (np.arange(24) * 5) % cfg.vocab_size
    ref = M.Model(cfg, W).forward(toks, logits="all")
    for be in ("w4a16", "w4a8"):
        q = M.Model(cfg, W, backend=be, **kw)
        assert _rel(q.forward(toks, logits="all"), ref) < 0.1
        if "quantize_lm_head" in kw:
            assert isinstance(q.lm_head, M.FP32Linear)
        else:
            assert (q.lm_head.wq.zeros is not None) == kw.get("asym", False)


def test_quantized_kv_decode_matches_full(tiny):
    cfg, W = tiny
    m = M.Model(cfg, W, backend="w4a8", threads=2)
    toks = (np.arange(20) * 3) % cfg.vocab_size
    full = m.forward(toks, logits="all")
    c = m.new_cache(32)
    inc = [m.forward(toks[:12], c, logits="all")] + [m.forward([t], c)[None] for t in toks[12:]]
    # w4a8 quantizes activations per token, so a token's result does not depend on M
    assert _rel(np.concatenate(inc), full) < 1e-4


def _write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))


def test_load_hf_dir_bf16(tmp_path, tiny):
    cfg, W = tiny
    bf16 = {n: (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16) for n, a in W.items()}
    _write_safetensors(tmp_path / "model.safetensors", {n: ("BF16", a) for n, a in bf16.items()})
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size, "num_hidden_layers": cfg.num_layers,
        "num_attention_heads": cfg.num_heads, "num_key_value_heads": cfg.num_kv_heads,
        "rms_norm_eps": cfg.rms_norm_eps, "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": True, "max_position_embeddings": cfg.max_position}))
    m = M.load(tmp_path)
    assert m.cfg.qkv_bias and m.cfg.head_dim == cfg.head_dim
    widened = {n: (a.astype(np.uint32) << 16).view(np.float32) for n, a in bf16.items()}
    toks = [1, 2, 3, 4]
    ref = M.Model(cfg, widened).forward(toks, logits="all")
    np.testing.assert_array_equal(m.forward(toks, logits="all"), ref)
