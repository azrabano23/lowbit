"""Pure-numpy reference GEMMs (the ground truth the C kernels are tested against)."""
from __future__ import annotations

import numpy as np

from .formats import GROUP, Int4Weights, Int8Acts, MXFP4Weights, quantize_act_int8


def gemm_w4a16_ref(x: np.ndarray, w: Int4Weights) -> np.ndarray:
    """Y = X @ dequant(W).T in float64 (returned as float64)."""
    return np.asarray(x, np.float64) @ w.dequantize().astype(np.float64).T


def gemm_mxfp4_ref(x: np.ndarray, w: MXFP4Weights) -> np.ndarray:
    return np.asarray(x, np.float64) @ w.dequantize().astype(np.float64).T


def w4a8_group_acc_ref(a: Int8Acts, w: Int4Weights) -> np.ndarray:
    """Exact integer accumulators acc[m, n, g] = sum_{k in g} (q[n,k] - z[n,g]) * xq[m,k]."""
    m, kp = a.xq.shape
    n, g = w.n, kp // GROUP
    q = w.codes().astype(np.int64).reshape(n, g, GROUP)
    z = np.full((n, g), 8, np.int64) if w.zeros is None else w.zeros.astype(np.int64)
    qc = q - z[:, :, None]
    xa = a.xq.astype(np.int64).reshape(m, g, GROUP)
    return np.einsum("mgk,ngk->mng", xa, qc)


def gemm_w4a8_ref(x: np.ndarray, w: Int4Weights, acts: Int8Acts | None = None) -> np.ndarray:
    """Y[m,n] = xs[m] * sum_g s[n,g] * acc[m,n,g]  in float64."""
    a = quantize_act_int8(x, w.kp) if acts is None else acts
    acc = w4a8_group_acc_ref(a, w).astype(np.float64)
    y = np.einsum("mng,ng->mn", acc, w.scales.astype(np.float64))
    return y * a.scale.astype(np.float64)[:, None]


def dequant_then_matmul(x: np.ndarray, w) -> np.ndarray:
    """The 'naive' baseline: materialize fp32 W, then BLAS sgemm."""
    return np.asarray(x, np.float32) @ w.dequantize().T
