"""Quantizers and packed weight containers (pure numpy).

Shapes follow nn.Linear: a weight matrix W is [N, K] (N outputs, K inputs)
and the GEMM is Y[M, N] = X[M, K] @ W.T.

Constraints (enforced here, relied on by the C kernels):
  * any M >= 1, N >= 1, K >= 1.
  * K is padded up to a multiple of the block size (128 for int4, 32 for
    MXFP4). A partial last group is quantized from its real elements only;
    padded codes decode to exactly 0 and the kernels see zero-padded X, so
    padding never changes the result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUP = 128
MXBLOCK = 32

# OCP MX E2M1 code -> value (bit 3 = sign, bits 2..1 = exponent, bit 0 = mantissa)
E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
E2M1_EMAX = 2  # exponent of the largest normal (6 = 1.5 * 2^2)


def pad_to(k: int, block: int) -> int:
    return -(-k // block) * block


# ---------------------------------------------------------------- nibble packing

def pack_nibbles(codes: np.ndarray, block: int) -> np.ndarray:
    """Pack uint8 codes [N, Kp] (values 0..15) into [N, Kp/2] bytes.

    Within each block of `block` elements, byte j holds element j in the low
    nibble and element j + block/2 in the high nibble. (block=128 for int4,
    32 for MXFP4 - the same "j / j+half" split ggml's Q4_0 uses per 32.)
    """
    n, kp = codes.shape
    assert kp % block == 0
    c = codes.reshape(n, kp // block, 2, block // 2).astype(np.uint8)
    return (c[:, :, 0, :] | (c[:, :, 1, :] << 4)).reshape(n, kp // 2)


def unpack_nibbles(packed: np.ndarray, block: int) -> np.ndarray:
    n, half = packed.shape
    kp = half * 2
    p = packed.reshape(n, kp // block, block // 2)
    return np.stack([p & 15, p >> 4], axis=2).reshape(n, kp)


# ---------------------------------------------------------------- int4 (g128)

@dataclass
class Int4Weights:
    """int4 weights, group size 128, fp32 per-group scale, optional uint8 zero-point.

    value[n, k] = (q[n, k] - z[n, g]) * s[n, g], g = k // 128, z = 8 if symmetric.
    """
    qw: np.ndarray            # uint8 [N, Kp/2] packed codes
    scales: np.ndarray        # float32 [N, G]
    zeros: np.ndarray | None  # uint8 [N, G] or None (symmetric, z = 8)
    k: int                    # logical K (before padding)

    @property
    def n(self) -> int:
        return self.qw.shape[0]

    @property
    def kp(self) -> int:
        return self.qw.shape[1] * 2

    @property
    def nbytes(self) -> int:
        return self.qw.nbytes + self.scales.nbytes + (0 if self.zeros is None else self.zeros.nbytes)

    def codes(self) -> np.ndarray:
        return unpack_nibbles(self.qw, GROUP)

    def dequantize(self) -> np.ndarray:
        """[N, K] float32 (only for references/baselines; kernels never do this)."""
        q = self.codes().astype(np.float32).reshape(self.n, -1, GROUP)
        z = np.float32(8.0) if self.zeros is None else self.zeros.astype(np.float32)[:, :, None]
        w = (q - z) * self.scales[:, :, None]
        return w.reshape(self.n, self.kp)[:, : self.k]


def quantize_int4(w: np.ndarray, asym: bool = False) -> Int4Weights:
    """Round-to-nearest int4 quantization with group size 128.

    symmetric: s = absmax / 7, q = clip(rint(w / s), -8, 7) + 8   (z = 8)
    asymmetric: s = (max - min) / 15, z = clip(rint(-min / s), 0, 15),
                q = clip(rint(w / s) + z, 0, 15)
    """
    w = np.asarray(w, dtype=np.float32)
    n, k = w.shape
    kp = pad_to(k, GROUP)
    g = kp // GROUP
    wp = np.zeros((n, kp), np.float32)
    wp[:, :k] = w
    valid = np.zeros(kp, bool)
    valid[:k] = True
    wg = wp.reshape(n, g, GROUP)
    vg = valid.reshape(g, GROUP)[None]
    if asym:
        wmax = np.where(vg, wg, -np.inf).max(axis=2)
        wmin = np.where(vg, wg, np.inf).min(axis=2)
        wmax = np.maximum(wmax, 0.0)  # keep 0 representable exactly
        wmin = np.minimum(wmin, 0.0)
        s = ((wmax - wmin) / np.float32(15)).astype(np.float32)
        s = np.where(s > 0, s, np.float32(1.0)).astype(np.float32)
        z = np.clip(np.rint(-wmin / s), 0, 15).astype(np.uint8)
        q = np.clip(np.rint(wg / s[:, :, None]) + z[:, :, None], 0, 15)
        q = np.where(vg, q, z[:, :, None])  # padding decodes to exactly 0
        zeros = z
    else:
        amax = np.abs(wg).max(axis=2)
        s = (amax / np.float32(7)).astype(np.float32)
        s = np.where(s > 0, s, np.float32(1.0)).astype(np.float32)
        q = np.clip(np.rint(wg / s[:, :, None]), -8, 7) + 8
        q = np.where(vg, q, 8)
        zeros = None
    codes = q.astype(np.uint8).reshape(n, kp)
    return Int4Weights(pack_nibbles(codes, GROUP), s.astype(np.float32), zeros, k)


# ---------------------------------------------------------------- MXFP4

@dataclass
class MXFP4Weights:
    """OCP MXFP4: E2M1 elements, blocks of 32, shared E8M0 scale 2^(e - 127)."""
    qw: np.ndarray    # uint8 [N, Kp/2] packed E2M1 codes
    e8m0: np.ndarray  # uint8 [N, Kp/32] biased exponents
    k: int

    @property
    def n(self) -> int:
        return self.qw.shape[0]

    @property
    def kp(self) -> int:
        return self.qw.shape[1] * 2

    @property
    def nbytes(self) -> int:
        return self.qw.nbytes + self.e8m0.nbytes

    def codes(self) -> np.ndarray:
        return unpack_nibbles(self.qw, MXBLOCK)

    def dequantize(self) -> np.ndarray:
        v = E2M1_VALUES[self.codes()].reshape(self.n, -1, MXBLOCK)
        scale = e8m0_to_float(self.e8m0)
        return (v * scale[:, :, None]).reshape(self.n, self.kp)[:, : self.k]


def e8m0_to_float(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=np.uint8)
    ex = np.where(e == 255, 0, e.astype(np.int32) - 127)
    out = np.ldexp(np.float32(1.0), ex).astype(np.float32)
    return np.where(e == 255, np.float32(np.nan), out).astype(np.float32)


def fp32_to_e2m1(v: np.ndarray) -> np.ndarray:
    """Round fp32 (already divided by the block scale) to E2M1 codes.

    Round-to-nearest, ties-to-even (on the mantissa bit), saturating to +-6
    as the OCP MX spec prescribes for FP4.
    """
    a = np.abs(np.asarray(v, np.float32))
    # midpoints between consecutive magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6.
    # tie goes to the even code: down at .25, 1.25, 2.5, 5; up at .75, 1.75, 3.5
    code = ((a > 0.25).astype(np.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75)
            + (a > 2.5) + (a >= 3.5) + (a > 5.0))
    sign = (np.signbit(v) & (code > 0)).astype(np.uint8) << 3
    return (code | sign).astype(np.uint8)


def quantize_mxfp4(w: np.ndarray) -> MXFP4Weights:
    """OCP MX v1.0 quantization: shared exponent = floor(log2(amax)) - emax_elem."""
    w = np.asarray(w, dtype=np.float32)
    n, k = w.shape
    kp = pad_to(k, MXBLOCK)
    wp = np.zeros((n, kp), np.float32)
    wp[:, :k] = w
    wb = wp.reshape(n, kp // MXBLOCK, MXBLOCK)
    amax = np.abs(wb).max(axis=2)
    with np.errstate(divide="ignore"):
        shared = np.floor(np.log2(amax, where=amax > 0, out=np.zeros_like(amax))) - E2M1_EMAX
    shared = np.where(amax > 0, shared, 0)
    e = np.clip(shared + 127, 0, 254).astype(np.uint8)
    scale = e8m0_to_float(e)
    codes = fp32_to_e2m1(wb / scale[:, :, None]).reshape(n, kp)
    return MXFP4Weights(pack_nibbles(codes, MXBLOCK), e, k)


# ---------------------------------------------------------------- activations

@dataclass
class Int8Acts:
    xq: np.ndarray    # int8 [M, Kp]
    scale: np.ndarray  # float32 [M] (per token)
    gsum: np.ndarray  # int32 [M, Kp/128]  per-group sum of codes

    def dequantize(self) -> np.ndarray:
        return self.xq.astype(np.float32) * self.scale[:, None]


def pad_acts(x: np.ndarray, kp: int) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape[1] == kp:
        return x
    out = np.zeros((x.shape[0], kp), np.float32)
    out[:, : x.shape[1]] = x
    return out


def quantize_act_int8(x: np.ndarray, kp: int | None = None) -> Int8Acts:
    """Per-token symmetric int8: s = absmax/127, q = clip(rint(x/s), -127, 127).

    Bit-identical to the C quantizer (lb_quant_act_int8) used inside the W4A8 kernel.
    """
    x = np.asarray(x, np.float32)
    kp = pad_to(x.shape[1], GROUP) if kp is None else kp
    xp = pad_acts(x, kp)
    amax = np.abs(xp).max(axis=1)
    s = np.where(amax > 0, amax / np.float32(127), np.float32(1)).astype(np.float32)
    q = np.clip(np.rint(xp / s[:, None]), -127, 127).astype(np.int8)
    gsum = q.reshape(q.shape[0], -1, GROUP).astype(np.int32).sum(axis=2).astype(np.int32)
    return Int8Acts(q, s, gsum)


# ---------------------------------------------------------------- AMX repack

@dataclass
class AMXInt4Weights:
    """int4 g128 weights repacked once for the AMX tile kernel (see kern_amx.c).

    Per block of 16 output rows and group of 128 k: 1 KiB where byte (r, 4c+i)
    holds code(n0+c, 128g+4r+i) (low nibble) and code(n0+c, 128g+64+4r+i)
    (high nibble) - the TDPBSUD "B" layout. Per-(block, group) scale vectors
    s[16] and s*z[16] are stored contiguously so the kernel needs no gathers.
    """
    qw: np.ndarray      # uint8 [NB * G * 1024]
    scales: np.ndarray  # float32 [NB, G, 16]
    sz: np.ndarray      # float32 [NB, G, 16]  = s * z
    zeros: np.ndarray   # uint8 [NB, G, 16]
    n: int
    k: int
    kp: int

    @property
    def nbytes(self) -> int:
        return self.qw.nbytes + self.scales.nbytes + self.sz.nbytes


def pack_amx(w: Int4Weights) -> AMXInt4Weights:
    n, kp = w.n, w.kp
    g = kp // GROUP
    nb = -(-n // 16)
    codes = np.zeros((nb * 16, kp), np.uint8)
    codes[:n] = w.codes()
    z = np.full((nb * 16, g), 8, np.uint8)
    if w.zeros is not None:
        z[:n] = w.zeros
    codes[n:] = np.repeat(z[n:], GROUP, axis=1)  # padded rows decode to 0
    s = np.zeros((nb * 16, g), np.float32)
    s[:n] = w.scales
    # [nb, c, g, half, r, i] -> [nb, g, r, c, i, half]
    c6 = codes.reshape(nb, 16, g, 2, 16, 4).transpose(0, 2, 4, 1, 5, 3)
    packed = (c6[..., 0] | (c6[..., 1] << 4)).astype(np.uint8)
    sb = np.ascontiguousarray(s.reshape(nb, 16, g).transpose(0, 2, 1))
    zb = np.ascontiguousarray(z.reshape(nb, 16, g).transpose(0, 2, 1))
    return AMXInt4Weights(np.ascontiguousarray(packed).reshape(-1), sb,
                          (sb * zb.astype(np.float32)).astype(np.float32), zb, n, w.k, kp)
