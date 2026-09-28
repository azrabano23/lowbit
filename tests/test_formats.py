import numpy as np
import pytest

from lowbit import formats as F


@pytest.mark.parametrize("block", [32, 128])
def test_nibble_pack_roundtrip(rng, block):
    codes = rng.integers(0, 16, size=(5, 3 * block), dtype=np.uint8)
    packed = F.pack_nibbles(codes, block)
    assert packed.shape == (5, 3 * block // 2)
    np.testing.assert_array_equal(F.unpack_nibbles(packed, block), codes)
    # documented layout: byte j = elem j (low) | elem j + block/2 (high)
    assert packed[0, 0] == codes[0, 0] | (codes[0, block // 2] << 4)


@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("k", [128, 300, 1, 257])
def test_int4_roundtrip_error_bound(rng, asym, k):
    w = rng.standard_normal((7, k)).astype(np.float32) * 3
    q = F.quantize_int4(w, asym=asym)
    assert q.kp % 128 == 0 and q.kp >= k
    wd = q.dequantize()
    assert wd.shape == w.shape
    s = np.repeat(q.scales, 128, axis=1)[:, :k]
    # round-to-nearest: error at most half a step (plus fp slack)
    assert np.all(np.abs(wd - w) <= 0.5 * s * (1 + 1e-5) + 1e-7)
    codes = q.codes()
    assert codes.max() <= 15
    # padded tail decodes to exactly zero
    zeros = 8 if q.zeros is None else q.zeros[:, -1:]
    np.testing.assert_array_equal(codes[:, k:], np.broadcast_to(zeros, codes[:, k:].shape))


def test_int4_exact_on_grid(rng):
    s = np.float32(0.25)
    codes = rng.integers(-7, 8, size=(4, 256))
    codes[:, 0] = 7  # pin the absmax so the scale is exactly s
    codes[:, 128] = -7
    w = (codes * s).astype(np.float32)
    q = F.quantize_int4(w)
    np.testing.assert_array_equal(q.dequantize(), w)


def test_int4_constant_and_zero_groups():
    w = np.zeros((2, 128), np.float32)
    w[1] = 5.0
    for asym in (False, True):
        q = F.quantize_int4(w, asym=asym)
        np.testing.assert_allclose(q.dequantize(), w, rtol=0, atol=1e-6)


def test_e2m1_table_and_rounding():
    vals = F.E2M1_VALUES
    codes = np.arange(16, dtype=np.uint8)
    # every representable value maps back to itself (canonical +0 for -0)
    c = F.fp32_to_e2m1(vals)
    np.testing.assert_array_equal(vals[c], vals)
    # ties go to the even mantissa code
    ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
    expect = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], np.float32)
    np.testing.assert_array_equal(vals[F.fp32_to_e2m1(ties)], expect)
    np.testing.assert_array_equal(vals[F.fp32_to_e2m1(-ties)], -expect)
    # saturation
    np.testing.assert_array_equal(vals[F.fp32_to_e2m1(np.array([7.9, -100.0], np.float32))],
                                  [6.0, -6.0])
    assert codes.dtype == np.uint8


def test_e8m0_decode():
    e = np.array([0, 1, 126, 127, 128, 254, 255], np.uint8)
    f = F.e8m0_to_float(e)
    np.testing.assert_array_equal(f[:6], np.array([2.0**-127, 2.0**-126, 0.5, 1.0, 2.0, 2.0**127],
                                                  np.float32))
    assert np.isnan(f[6])


@pytest.mark.parametrize("k", [32, 100, 4096])
def test_mxfp4_roundtrip(rng, k):
    w = rng.standard_normal((6, k)).astype(np.float32)
    q = F.quantize_mxfp4(w)
    assert q.kp % 32 == 0
    wd = q.dequantize()
    scale = np.repeat(F.e8m0_to_float(q.e8m0), 32, axis=1)[:, :k]
    # the largest step of E2M1 is 2 (between 4 and 6); saturation at 6*scale
    # can clip values in [6, 8)*scale, so the bound is 2 * scale
    assert np.all(np.abs(wd - w) <= 2.0 * scale)
    # shared exponent follows the spec: floor(log2(amax)) - 2
    blk = np.pad(w, ((0, 0), (0, q.kp - k))).reshape(6, -1, 32)
    amax = np.abs(blk).max(axis=2)
    np.testing.assert_array_equal(q.e8m0.astype(int) - 127, np.floor(np.log2(amax)) - 2)


def test_mxfp4_exact_on_grid(rng):
    codes = rng.integers(0, 16, size=(3, 64)).astype(np.uint8)
    codes[:, 0] = 7   # amax 6 -> shared exponent floor(log2 6) - 2 = 0
    codes[:, 32] = 7
    w = F.E2M1_VALUES[codes] * np.float32(4.0)
    q = F.quantize_mxfp4(w)
    np.testing.assert_array_equal(q.dequantize(), w)
    np.testing.assert_array_equal(q.e8m0, 129)


def test_act_quant(rng):
    x = rng.standard_normal((4, 200)).astype(np.float32)
    a = F.quantize_act_int8(x)
    assert a.xq.shape == (4, 256) and a.gsum.shape == (4, 2)
    assert np.all(np.abs(a.xq) <= 127)
    np.testing.assert_array_equal(a.xq[:, 200:], 0)
    np.testing.assert_array_equal(a.gsum, a.xq.astype(np.int32).reshape(4, 2, 128).sum(-1))
    assert np.all(np.abs(a.dequantize()[:, :200] - x) <= 0.5 * a.scale[:, None] * (1 + 1e-6))
