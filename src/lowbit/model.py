"""A minimal decoder-only LLM forward pass in numpy (Qwen2 / Llama architecture).

Components: token embedding, RMSNorm, rotary position embedding (the
"rotate_half" convention of the Hugging Face checkpoints), grouped-query
attention with a KV cache, SwiGLU MLP and a tied or untied LM head.

Every linear layer goes through a pluggable backend chosen at load time:

  fp32    numpy ``x @ W.T`` on fp32 weights (OpenBLAS sgemm) - the reference
  w4a16   lowbit int4 g128 weights, fp32 activations   (``gemm_w4a16``)
  w4a8    same int4 weights, int8 per-token activations (``gemm_w4a8``)
  mxfp4   OCP MXFP4 (E2M1, block 32, E8M0 scale)        (``gemm_mxfp4``)

Weights are quantized once, round-to-nearest, with the library's quantizers
(group 128 for int4 - symmetric absmax/7 by default, optionally asymmetric or
ggml-Q4_0-style signed-max scales - and block 32 for MXFP4). The q/k/v projections and the
gate/up projections are concatenated along N into one GEMM each (quantization
is per output row, so this does not change any quantized value). Norm weights,
biases, the embedding table used for lookup and the KV cache stay fp32.

Nothing here needs torch: weights are read straight from ``model.safetensors``
(bf16/fp16/fp32) and the tokenizer (optional) is the ``tokenizers`` package.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BACKENDS = ("fp32", "w4a16", "w4a8", "mxfp4")


# ---------------------------------------------------------------- config

@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    qkv_bias: bool = False
    max_position: int = 4096
    name: str = "custom"

    @classmethod
    def from_hf(cls, cfg: dict, name: str = "custom") -> "ModelConfig":
        mt = cfg.get("model_type", "llama")
        if mt not in ("qwen2", "llama", "mistral"):
            raise ValueError(f"unsupported model_type {mt!r} (qwen2/llama/mistral only)")
        if cfg.get("rope_scaling"):
            raise ValueError("rope_scaling is not implemented")
        h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
        return cls(vocab_size=cfg["vocab_size"], hidden_size=h,
                   intermediate_size=cfg["intermediate_size"],
                   num_layers=cfg["num_hidden_layers"], num_heads=nh,
                   num_kv_heads=cfg.get("num_key_value_heads", nh),
                   head_dim=cfg.get("head_dim") or h // nh,
                   rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
                   rope_theta=cfg.get("rope_theta", 10000.0),
                   tie_word_embeddings=cfg.get("tie_word_embeddings", False),
                   qkv_bias=(mt == "qwen2") or cfg.get("attention_bias", False),
                   max_position=cfg.get("max_position_embeddings", 4096), name=name)


# ---------------------------------------------------------------- safetensors

_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def load_safetensors(path: Path | str) -> dict[str, np.ndarray]:
    """Read a .safetensors file into fp32 numpy arrays (bf16 widened exactly)."""
    path = Path(path)
    with open(path, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(hlen))
    base = 8 + hlen
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = info["dtype"]
        if dt not in _ST_DTYPES:
            raise ValueError(f"{name}: unsupported dtype {dt}")
        a, b = info["data_offsets"]
        raw = np.frombuffer(mm[base + a: base + b], dtype=_ST_DTYPES[dt])
        if dt == "BF16":
            arr = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = raw.astype(np.float32)
        out[name] = arr.reshape(info["shape"])
    return out


# ---------------------------------------------------------------- linear backends

class Linear:
    """y = x @ W.T (+ b). Subclasses hold W in some format."""
    kind = "?"

    def __init__(self, w: np.ndarray, b: np.ndarray | None, threads: int | None):
        self.n, self.k = w.shape
        self.b = None if b is None else np.asarray(b, np.float32)
        self.threads = threads

    def mm(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self.mm(x)
        if self.b is not None:
            y += self.b
        return y


class FP32Linear(Linear):
    kind = "fp32"

    def __init__(self, w, b=None, threads=None, **_):
        super().__init__(w, b, threads)
        self.w = np.ascontiguousarray(w, np.float32)
        self.nbytes = self.w.nbytes

    def mm(self, x):
        return x @ self.w.T


class _LowbitLinear(Linear):
    def mm(self, x):
        return self._gemm(x, self.wq, self.params, threads=self.threads)


class W4A16Linear(_LowbitLinear):
    kind = "w4a16"

    def __init__(self, w, b=None, threads=None, asym=False, params=None, int4_scale="absmax",
                 **_):
        from .formats import quantize_int4
        from .kernels import gemm_w4a16
        super().__init__(w, b, threads)
        self._gemm = gemm_w4a16
        self.wq = quantize_int4(w, asym=asym, scale=int4_scale)
        self.params = params
        self.nbytes = self.wq.nbytes


class W4A8Linear(_LowbitLinear):
    kind = "w4a8"

    def __init__(self, w, b=None, threads=None, asym=False, params=None, int4_scale="absmax",
                 **_):
        from .formats import quantize_int4
        from .kernels import gemm_w4a8
        super().__init__(w, b, threads)
        self._gemm = gemm_w4a8
        self.wq = quantize_int4(w, asym=asym, scale=int4_scale)
        self.params = params
        self.nbytes = self.wq.nbytes


class MXFP4Linear(_LowbitLinear):
    kind = "mxfp4"

    def __init__(self, w, b=None, threads=None, params=None, **_):
        from .formats import quantize_mxfp4
        from .kernels import gemm_mxfp4
        super().__init__(w, b, threads)
        self._gemm = gemm_mxfp4
        self.wq = quantize_mxfp4(w)
        self.params = params
        self.nbytes = self.wq.nbytes


LINEARS = {"fp32": FP32Linear, "w4a16": W4A16Linear, "w4a8": W4A8Linear, "mxfp4": MXFP4Linear}


# ---------------------------------------------------------------- building blocks

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + np.float32(eps))).astype(np.float32) * w


def rope_tables(head_dim: int, n_pos: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    ang = np.arange(n_pos, dtype=np.float64)[:, None] * inv[None]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x [T, H, D]; cos/sin [T, D/2]. HF 'rotate_half' convention (halves, not pairs)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[:, None, :], sin[:, None, :]
    out = np.empty_like(x)
    out[..., :h] = x1 * c - x2 * s
    out[..., h:] = x2 * c + x1 * s
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


@dataclass
class KVCache:
    k: list  # per layer [KV, S_max, D]
    v: list
    length: int = 0

    @classmethod
    def empty(cls, cfg: ModelConfig, max_len: int) -> "KVCache":
        shape = (cfg.num_kv_heads, max_len, cfg.head_dim)
        return cls([np.zeros(shape, np.float32) for _ in range(cfg.num_layers)],
                   [np.zeros(shape, np.float32) for _ in range(cfg.num_layers)])

    @property
    def capacity(self) -> int:
        return self.k[0].shape[1]


@dataclass
class Layer:
    ln1: np.ndarray
    qkv: Linear
    o: Linear
    ln2: np.ndarray
    gate_up: Linear
    down: Linear


@dataclass
class Profile:
    """Wall time per section, accumulated across forward calls."""
    t: dict = field(default_factory=lambda: defaultdict(float))
    enabled: bool = True

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] += time.perf_counter() - t0

    def reset(self):
        self.t = defaultdict(float)


# ---------------------------------------------------------------- model

_CTL: list = []


def _threadpool_controller():
    """One cached threadpoolctl controller (building one per call costs ~0.15 ms)."""
    if not _CTL:
        try:
            from threadpoolctl import ThreadpoolController
            _CTL.append(ThreadpoolController())
        except ImportError:  # pragma: no cover
            _CTL.append(None)
    return _CTL[0]


class Model:
    def __init__(self, cfg: ModelConfig, weights: dict[str, np.ndarray], backend: str = "fp32",
                 threads: int | None = None, asym: bool = False, quantize_lm_head: bool = True,
                 params=None, int4_scale: str = "absmax"):
        if backend not in LINEARS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.cfg, self.backend, self.threads = cfg, backend, threads
        self.asym, self.quantize_lm_head, self.int4_scale = asym, quantize_lm_head, int4_scale
        lin = LINEARS[backend]
        kw = dict(threads=threads, asym=asym, params=params, int4_scale=int4_scale)
        W = weights
        g = lambda n: np.asarray(W[n], np.float32)  # noqa: E731
        self.embed = np.ascontiguousarray(g("model.embed_tokens.weight"))
        self.layers: list[Layer] = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            wqkv = np.concatenate([g(a + "q_proj.weight"), g(a + "k_proj.weight"),
                                   g(a + "v_proj.weight")], axis=0)
            bqkv = (np.concatenate([g(a + "q_proj.bias"), g(a + "k_proj.bias"),
                                    g(a + "v_proj.bias")]) if cfg.qkv_bias else None)
            wgu = np.concatenate([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")], 0)
            self.layers.append(Layer(
                ln1=g(p + "input_layernorm.weight"), qkv=lin(wqkv, bqkv, **kw),
                o=lin(g(a + "o_proj.weight"), None, **kw),
                ln2=g(p + "post_attention_layernorm.weight"), gate_up=lin(wgu, None, **kw),
                down=lin(g(p + "mlp.down_proj.weight"), None, **kw)))
        self.norm = g("model.norm.weight")
        head = self.embed if cfg.tie_word_embeddings or "lm_head.weight" not in W \
            else g("lm_head.weight")
        self.lm_head = (lin if quantize_lm_head else FP32Linear)(head, None, **kw)
        self.cos, self.sin = rope_tables(cfg.head_dim, cfg.max_position, cfg.rope_theta)
        self.prof = Profile()

    # ------------------------------------------------------------ bookkeeping

    @property
    def linear_weight_bytes(self) -> int:
        return sum(lin.nbytes for lin in self.linears())

    def linears(self) -> list[Linear]:
        return [x for L in self.layers for x in (L.qkv, L.o, L.gate_up, L.down)] + [self.lm_head]

    def set_threads(self, threads: int | None) -> None:
        self.threads = threads
        for lin in self.linears():
            lin.threads = threads

    def new_cache(self, max_len: int) -> KVCache:
        return KVCache.empty(self.cfg, max_len)

    @contextmanager
    def blas_threads(self):
        """Thread count for OpenBLAS (the fp32 GEMMs and the attention matmuls).

        fp32 backend: the model's thread count. lowbit backends: 1 thread,
        because OpenBLAS's pthreads keep spinning after each call and fight the
        kernels' OpenMP threads for the same cores (measured: 64-token prefill
        3.5x slower at 4 threads when both pools had 4 threads).
        """
        n = self.threads if self.backend == "fp32" else 1
        if n is None:
            yield
            return
        ctl = _threadpool_controller()
        if ctl is None:  # pragma: no cover - threadpoolctl not installed
            yield
            return
        with ctl.limit(limits=n, user_api="blas"):
            yield

    # ------------------------------------------------------------ forward

    def _attention(self, q, k, v, cache: KVCache | None, li: int, pos: int):
        """q [T, H, D], k/v [T, KV, D] (after RoPE) -> [T, H*D]."""
        cfg = self.cfg
        T, H, D = q.shape
        KV = cfg.num_kv_heads
        G = H // KV
        if cache is not None:
            cache.k[li][:, pos:pos + T] = k.transpose(1, 0, 2)
            cache.v[li][:, pos:pos + T] = v.transpose(1, 0, 2)
            S = pos + T
            ks, vs = cache.k[li][:, :S], cache.v[li][:, :S]
        else:
            S = T
            ks, vs = k.transpose(1, 0, 2), v.transpose(1, 0, 2)
        # [KV, G*T, D]: query heads sharing a KV head are batched into one matmul
        qh = q.reshape(T, KV, G, D).transpose(1, 2, 0, 3).reshape(KV, G * T, D)
        s = (qh @ ks.transpose(0, 2, 1)).reshape(KV, G, T, S)
        s *= np.float32(1.0 / np.sqrt(D))
        if T > 1:  # causal: query t (absolute pos+t) sees keys 0..pos+t
            mask = np.arange(S)[None, :] > (pos + np.arange(T))[:, None]
            s[:, :, mask] = -np.inf
        s -= s.max(axis=-1, keepdims=True)
        np.exp(s, out=s)
        s /= s.sum(axis=-1, keepdims=True)
        o = s.reshape(KV, G * T, S) @ vs  # [KV, G*T, D]
        return o.reshape(KV, G, T, D).transpose(2, 0, 1, 3).reshape(T, H * D)

    def forward(self, tokens, cache: KVCache | None = None, logits: str = "last",
                head_chunk: int = 256) -> np.ndarray:
        """Run tokens [T] at positions cache.length.. (0 without a cache).

        logits="last" returns [V] for the last position, "all" returns [T, V],
        "none" returns the final hidden states [T, hidden].
        """
        cfg, P = self.cfg, self.prof
        tokens = np.asarray(tokens, np.int64).reshape(-1)
        T = tokens.shape[0]
        pos = cache.length if cache is not None else 0
        if cache is not None and pos + T > cache.capacity:
            raise ValueError(f"KV cache full ({pos}+{T} > {cache.capacity})")
        H, KV, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        with self.blas_threads():
            with P.section("other"):
                x = self.embed[tokens]
                cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
            for li, L in enumerate(self.layers):
                with P.section("other"):
                    h = rms_norm(x, L.ln1, cfg.rms_norm_eps)
                with P.section("linear"):
                    qkv = L.qkv(h)
                with P.section("attention"):
                    q = apply_rope(qkv[:, :H * D].reshape(T, H, D), cos, sin)
                    k = apply_rope(qkv[:, H * D:(H + KV) * D].reshape(T, KV, D), cos, sin)
                    v = qkv[:, (H + KV) * D:].reshape(T, KV, D)
                    a = self._attention(q, k, v, cache, li, pos)
                with P.section("linear"):
                    y = L.o(a)
                with P.section("other"):
                    x = x + y
                    h = rms_norm(x, L.ln2, cfg.rms_norm_eps)
                with P.section("linear"):
                    gu = L.gate_up(h)
                with P.section("other"):
                    I = cfg.intermediate_size
                    m = silu(gu[:, :I]) * gu[:, I:]
                with P.section("linear"):
                    y = L.down(m)
                with P.section("other"):
                    x = x + y
            if cache is not None:
                cache.length = pos + T
            with P.section("other"):
                x = rms_norm(x, self.norm, cfg.rms_norm_eps)
            if logits == "none":
                return x
            if logits == "last":
                with P.section("lm_head"):
                    return self.lm_head(x[-1:])[0]
            out = np.empty((T, cfg.vocab_size), np.float32)
            with P.section("lm_head"):
                for i in range(0, T, head_chunk):
                    out[i:i + head_chunk] = self.lm_head(x[i:i + head_chunk])
            return out

    def generate(self, prompt, n_new: int, cache: KVCache | None = None,
                 timings: dict | None = None) -> list[int]:
        """Greedy decoding: prefill the prompt, then n_new tokens one at a time."""
        prompt = list(prompt)
        cache = cache or self.new_cache(len(prompt) + n_new)
        t0 = time.perf_counter()
        logit = self.forward(prompt, cache)
        t1 = time.perf_counter()
        out = []
        for i in range(n_new):
            tok = int(np.argmax(logit))
            out.append(tok)
            if i + 1 < n_new:
                logit = self.forward([tok], cache)
        t2 = time.perf_counter()
        if timings is not None:
            timings["prefill_s"] = t1 - t0
            # n_new tokens are emitted but only n_new - 1 decode forwards run
            timings["decode_s"] = t2 - t1
            timings["decode_forwards"] = max(0, n_new - 1)
        return out


# ---------------------------------------------------------------- loading

def load(model_dir: Path | str, backend: str = "fp32", threads: int | None = None,
         **kw) -> Model:
    d = Path(model_dir)
    cfg = ModelConfig.from_hf(json.loads((d / "config.json").read_text()), name=d.name)
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    W: dict[str, np.ndarray] = {}
    for f in files:
        W.update(load_safetensors(f))
    return Model(cfg, W, backend=backend, threads=threads, **kw)


def load_tokenizer(model_dir: Path | str):
    """HF fast tokenizer from tokenizer.json (needs the `tokenizers` package)."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))


# ---------------------------------------------------------------- tiny random models (tests)

def tiny_config(**over) -> ModelConfig:
    base = dict(vocab_size=97, hidden_size=256, intermediate_size=384, num_layers=2,
                num_heads=4, num_kv_heads=2, head_dim=64, rms_norm_eps=1e-6,
                rope_theta=10000.0, tie_word_embeddings=True, qkv_bias=True,
                max_position=256, name="tiny")
    base.update(over)
    return ModelConfig(**base)


def random_weights(cfg: ModelConfig, seed: int = 0, scale: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    r = lambda *s: (rng.standard_normal(s) * scale).astype(np.float32)  # noqa: E731
    h, i, D = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    W = {"model.embed_tokens.weight": r(cfg.vocab_size, h) * 20,
         "model.norm.weight": 1 + r(h)}
    for li in range(cfg.num_layers):
        p = f"model.layers.{li}."
        W[p + "input_layernorm.weight"] = 1 + r(h)
        W[p + "post_attention_layernorm.weight"] = 1 + r(h)
        for nm, n in (("q", cfg.num_heads * D), ("k", cfg.num_kv_heads * D),
                      ("v", cfg.num_kv_heads * D)):
            W[p + f"self_attn.{nm}_proj.weight"] = r(n, h)
            if cfg.qkv_bias:
                W[p + f"self_attn.{nm}_proj.bias"] = r(n)
        W[p + "self_attn.o_proj.weight"] = r(h, cfg.num_heads * D)
        W[p + "mlp.gate_proj.weight"] = r(i, h)
        W[p + "mlp.up_proj.weight"] = r(i, h)
        W[p + "mlp.down_proj.weight"] = r(h, i)
    if not cfg.tie_word_embeddings:
        W["lm_head.weight"] = r(cfg.vocab_size, h)
    return W
