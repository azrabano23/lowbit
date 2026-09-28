"""C kernels vs the numpy reference, for every ISA this CPU can run."""
import numpy as np
import pytest
from conftest import ISAS, assert_fp_close

import lowbit as lb
from lowbit import kernels as K
from lowbit import reference as R

# (M, N, K): odd sizes, K not a multiple of the group/block, N not a multiple
# of any micro-tile, M not a multiple of mr.
SHAPES = [(1, 1, 1), (1, 64, 128), (3, 5, 127), (7, 33, 129), (5, 37, 300),
          (17, 70, 1000), (2, 16, 4096 + 32)]
PARAMS = [
    K.Params(),
    K.Params(mr=1, nr=1, nb=8),
    K.Params(mr=4, nr=4, nb=16, kb=128, order=1),
    K.Params(mr=8, nr=2, nb=32, kb=256, order=0),
    K.Params(mr=2, nr=4, nb=4, kb=512, order=1),
]


def _problem(rng, m, n, k):
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k)).astype(np.float32)
    return x, w


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("asym", [False, True])
def test_w4a16(rng, isa, shape, asym):
    x, w = _problem(rng, *shape)
    W = lb.quantize_int4(w, asym=asym)
    ref = R.gemm_w4a16_ref(x, W)
    bound = np.abs(x).astype(np.float64) @ np.abs(W.dequantize()).T.astype(np.float64)
    for p in PARAMS:
        y = K.gemm_w4a16(x, W, p, isa=isa)
        assert y.shape == (shape[0], shape[1])
        assert_fp_close(y, ref, bound)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("shape", SHAPES)
def test_mxfp4(rng, isa, shape):
    x, w = _problem(rng, *shape)
    W = lb.quantize_mxfp4(w)
    ref = R.gemm_mxfp4_ref(x, W)
    bound = np.abs(x).astype(np.float64) @ np.abs(W.dequantize()).T.astype(np.float64)
    for p in PARAMS:
        assert_fp_close(K.gemm_mxfp4(x, W, p, isa=isa), ref, bound)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("asym", [False, True])
def test_w4a8_int32_accumulators_bit_exact(rng, isa, shape, asym):
    x, w = _problem(rng, *shape)
    W = lb.quantize_int4(w, asym=asym)
    a = lb.quantize_act_int8(x)
    acc = K.w4a8_group_acc(a, W, isa=isa)
    np.testing.assert_array_equal(acc, R.w4a8_group_acc_ref(a, W))


def test_w4a8_extreme_values_exact():
    # worst case for saturation / overflow: codes 15 and 0, activations +-127/-128
    m, n, k = 3, 4, 512
    W = lb.quantize_int4(np.zeros((n, k), np.float32), asym=True)
    codes = np.zeros((n, k), np.uint8)
    codes[0] = 15
    codes[1, ::2] = 15
    codes[2] = np.arange(k) % 16
    W.qw[:] = lb.formats.pack_nibbles(codes, 128)
    W.zeros[:] = np.array([0, 15, 7, 8], np.uint8)[:, None]
    xq = np.full((m, k), 127, np.int8)
    xq[1] = -128
    xq[2, ::2] = -128
    a = lb.Int8Acts(xq, np.ones(m, np.float32), xq.astype(np.int32).reshape(m, -1, 128).sum(-1).astype(np.int32))
    ref = R.w4a8_group_acc_ref(a, W)
    for isa in ISAS:
        np.testing.assert_array_equal(K.w4a8_group_acc(a, W, isa=isa), ref)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("asym", [False, True])
def test_w4a8_gemm(rng, isa, shape, asym):
    x, w = _problem(rng, *shape)
    W = lb.quantize_int4(w, asym=asym)
    a = lb.quantize_act_int8(x)
    ref = R.gemm_w4a8_ref(x, W, a)
    # The int32 lanes are exact; the float part (lane -> fp32, * s, sum) has
    # fp32 rounding relative to the lane magnitudes, i.e. to
    # sum_k q|a| + z|sum a| per group (not to |acc|, which may cancel).
    g = W.kp // 128
    q = W.codes().astype(np.float64).reshape(W.n, g, 128)
    z = np.full((W.n, g), 8.0) if W.zeros is None else W.zeros.astype(np.float64)
    aa = np.abs(a.xq.astype(np.float64)).reshape(-1, g, 128)
    mag = np.einsum("mgk,ngk->mng", aa, q) + z[None] * np.abs(a.gsum)[:, None, :]
    bound = np.einsum("mng,ng->mn", mag, W.scales.astype(np.float64)) * a.scale[:, None]
    for p in PARAMS:
        assert_fp_close(K.gemm_w4a8(x, W, p, isa=isa), ref, bound)
        assert_fp_close(K.gemm_w4a8_q(a, W, p, isa=isa), ref, bound)


@pytest.mark.parametrize("shape", [(1, 1), (3, 127), (4, 1000), (2, 4096)])
def test_c_act_quant_matches_numpy_exactly(rng, shape):
    x = rng.standard_normal(shape).astype(np.float32) * 3
    x[0, 0] = 0.5 * np.abs(x).max() / 127 * 3  # an exact .5 tie after scaling (roughly)
    kp = lb.formats.pad_to(shape[1], 128)
    a_c = K.quant_act_int8(x, kp)
    a_np = lb.quantize_act_int8(x)
    np.testing.assert_array_equal(a_c.xq, a_np.xq)
    np.testing.assert_array_equal(a_c.scale, a_np.scale)
    np.testing.assert_array_equal(a_c.gsum, a_np.gsum)


def test_zero_activation_row(rng):
    x = rng.standard_normal((3, 256)).astype(np.float32)
    x[1] = 0
    W = lb.quantize_int4(rng.standard_normal((8, 256)).astype(np.float32))
    for isa in ISAS:
        y = K.gemm_w4a8(x, W, isa=isa)
        assert np.all(y[1] == 0) and np.all(np.isfinite(y))


def test_isa_clamped_to_hardware():
    top = ISAS[-1]
    assert K.effective_isa("w4a16", "avx512") == top
    assert K.effective_isa("w4a16", "scalar") == "scalar"


def test_bad_tile_rejected(rng):
    if ISAS[-1] == "scalar":
        pytest.skip("scalar forces 1x1 tiles")
    W = lb.quantize_int4(rng.standard_normal((8, 128)).astype(np.float32))
    with pytest.raises(ValueError):
        K.gemm_w4a16(np.ones((1, 128), np.float32), W, K.Params(mr=3, nr=3))


amx = pytest.mark.skipif(not K.cpu_info()["amx"], reason="AMX-INT8 not available")


@amx
@pytest.mark.parametrize("shape", SHAPES + [(33, 20, 256)])
@pytest.mark.parametrize("asym", [False, True])
def test_amx_int32_accumulators_bit_exact(rng, shape, asym):
    x, w = _problem(rng, *shape)
    W = lb.quantize_int4(w, asym=asym)
    A = lb.formats.pack_amx(W)
    a = lb.quantize_act_int8(x)
    np.testing.assert_array_equal(K.w4a8_group_acc_amx(a, A), R.w4a8_group_acc_ref(a, W))


@amx
@pytest.mark.parametrize("shape", SHAPES + [(33, 20, 256)])
def test_amx_gemm(rng, shape):
    x, w = _problem(rng, *shape)
    W = lb.quantize_int4(w, asym=True)
    A = lb.formats.pack_amx(W)
    ref = R.gemm_w4a8_ref(x, W)
    a = lb.quantize_act_int8(x)
    g = W.kp // 128
    q = W.codes().astype(np.float64).reshape(W.n, g, 128)
    aa = np.abs(a.xq.astype(np.float64)).reshape(-1, g, 128)
    mag = np.einsum("mgk,ngk->mng", aa, q) + W.zeros.astype(np.float64)[None] * np.abs(a.gsum)[:, None, :]
    bound = np.einsum("mng,ng->mn", mag, W.scales.astype(np.float64)) * a.scale[:, None]
    for th in (1, 3):
        assert_fp_close(K.gemm_w4a8_amx(x, A, threads=th), ref, bound)


def test_amx_pack_layout(rng):
    W = lb.quantize_int4(rng.standard_normal((20, 256)).astype(np.float32))
    A = lb.formats.pack_amx(W)
    codes = W.codes()
    blk = A.qw.reshape(2, 2, 16, 16, 4)  # [nblock, group, r, c, i]
    # byte (r, 4c+i) of block 0, group 1 = code(c, 128 + 4r + i) | code(c, 192 + 4r + i) << 4
    r, c, i = 5, 7, 2
    assert blk[0, 1, r, c, i] == codes[c, 128 + 4 * r + i] | (codes[c, 192 + 4 * r + i] << 4)
    assert A.scales.shape == (2, 2, 16) and A.scales[1, 0, 3] == W.scales[19, 0]
