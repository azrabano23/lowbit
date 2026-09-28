"""ctypes bindings to the C kernels."""
from __future__ import annotations

import ctypes as C
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from ._build import build
from .formats import GROUP, AMXInt4Weights, Int4Weights, Int8Acts, MXFP4Weights, pad_acts

ISA_NAMES = {0: "scalar", 1: "avx2", 2: "avx512"}
ISA_IDS = {v: k for k, v in ISA_NAMES.items()}
KIND_IDS = {"w4a16": 0, "mxfp4": 1, "w4a8": 2}


class _Params(C.Structure):
    _fields_ = [(f, C.c_int) for f in ("isa", "mr", "nr", "nb", "kb", "order", "nthreads")]


@dataclass(frozen=True)
class Params:
    """Kernel tiling parameters (see lowbit.h). 0 means 'library default'."""
    mr: int = 0
    nr: int = 0
    nb: int = 0
    kb: int = 0
    order: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


TILES = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2)]

_f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_i8p = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")
_i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_vp = C.c_void_p
_int = C.c_int
_pp = C.POINTER(_Params)


@lru_cache(maxsize=1)
def lib() -> C.CDLL:
    so = C.CDLL(str(build()))
    sig = {
        "lb_version": ([], _int),
        "lb_cpu_isa": ([], _int),
        "lb_active_isa": ([], _int),
        "lb_has_vnni": ([], _int),
        "lb_has_openmp": ([], _int),
        "lb_max_threads": ([], _int),
        "lb_effective_isa": ([_int, _int], _int),
        "lb_gemm_w4a16": ([_f32p, _int, _int, _u8p, _f32p, _vp, _int, _f32p, _pp], _int),
        "lb_gemm_mxfp4": ([_f32p, _int, _int, _u8p, _u8p, _int, _f32p, _pp], _int),
        "lb_gemm_w4a8": ([_f32p, _int, _int, _u8p, _f32p, _vp, _int, _f32p, _pp], _int),
        "lb_gemm_w4a8_q": ([_i8p, _f32p, _i32p, _int, _int, _u8p, _f32p, _vp, _int, _f32p, _pp], _int),
        "lb_quant_act_int8": ([_f32p, _int, _int, _i8p, _f32p, _i32p], None),
        "lb_w4a8_group_acc": ([_i8p, _i32p, _int, _int, _u8p, _vp, _int, _i32p, _int], _int),
    }
    if hasattr(so, "lb_amx_available"):
        sig.update({
            "lb_amx_available": ([], _int),
            "lb_gemm_w4a8_amx": ([_f32p, _int, _int, _u8p, _f32p, _f32p, _int, _f32p, _int], _int),
            "lb_w4a8_group_acc_amx": ([_i8p, _i32p, _int, _int, _u8p, _u8p, _int, _i32p], _int),
        })
    for name, (args, res) in sig.items():
        fn = getattr(so, name)
        fn.argtypes = args
        fn.restype = res
    return so


def cpu_info() -> dict:
    L = lib()
    return {
        "cpu_isa": ISA_NAMES[L.lb_cpu_isa()],
        "active_isa": ISA_NAMES[L.lb_active_isa()],
        "vnni": bool(L.lb_has_vnni()),
        "openmp": bool(L.lb_has_openmp()),
        "max_threads": L.lb_max_threads(),
        "amx": amx_available(),
    }


def amx_available() -> bool:
    """AMX-INT8 usable: compiled in, CPUID says so, the OS granted tile state,
    and LOWBIT_ISA does not cap us below avx512."""
    L = lib()
    return (hasattr(L, "lb_amx_available") and L.lb_active_isa() >= 2
            and bool(L.lb_amx_available()))


def _isa_id(isa: str | int | None) -> int:
    if isa is None or isa == "auto":
        return -1
    return isa if isinstance(isa, int) else ISA_IDS[isa]


def effective_isa(kind: str, isa: str | int | None = None) -> str:
    return ISA_NAMES[lib().lb_effective_isa(KIND_IDS[kind], _isa_id(isa))]


_AUTO_MEMO: dict = {}


def auto_params(kind: str, m: int, n: int, k: int, isa=None, threads=None) -> Params:
    """Tuned parameters from the tune cache (see lowbit.tune), else library defaults."""
    key = (kind, m, n, k, isa, threads)
    if key not in _AUTO_MEMO:
        from .tune import lookup
        _AUTO_MEMO[key] = lookup(kind, m, n, k, isa, threads) or Params()
    return _AUTO_MEMO[key]


def _params(p, isa, threads: int | None, kind=None, shape=None) -> _Params:
    if isinstance(p, str):
        if p != "auto":
            raise ValueError(f"params must be a Params, None or 'auto', not {p!r}")
        p = auto_params(kind, *shape, isa=isa, threads=threads)
    p = p or Params()
    return _Params(_isa_id(isa), p.mr, p.nr, p.nb, p.kb, p.order, threads or 0)


def _zp(z: np.ndarray | None):
    return None if z is None else z.ctypes.data_as(C.c_void_p)


def _check(rc: int, what: str) -> int:
    if rc < 0:
        raise ValueError(f"{what} failed (rc={rc}): bad shape or tile not compiled for this ISA")
    return rc


class _Prepared:
    """Padded, contiguous X plus an output buffer (reusable across timing reps)."""

    def __init__(self, x: np.ndarray, kp: int, n: int):
        x = np.asarray(x, np.float32)
        if x.ndim == 1:
            x = x[None]
        self.x = pad_acts(x, kp)
        self.y = np.empty((x.shape[0], n), np.float32)


def gemm_w4a16(x, w: Int4Weights, params: Params | None = None, isa=None,
               threads: int | None = None, _prep: _Prepared | None = None) -> np.ndarray:
    pr = _prep or _Prepared(x, w.kp, w.n)
    rc = lib().lb_gemm_w4a16(pr.x, pr.x.shape[0], w.kp, w.qw, w.scales, _zp(w.zeros), w.n,
                             pr.y, C.byref(_params(params, isa, threads, "w4a16",
                                                   (pr.x.shape[0], w.n, w.k))))
    _check(rc, "lb_gemm_w4a16")
    return pr.y


def gemm_mxfp4(x, w: MXFP4Weights, params: Params | None = None, isa=None,
               threads: int | None = None, _prep: _Prepared | None = None) -> np.ndarray:
    pr = _prep or _Prepared(x, w.kp, w.n)
    rc = lib().lb_gemm_mxfp4(pr.x, pr.x.shape[0], w.kp, w.qw, w.e8m0, w.n, pr.y,
                             C.byref(_params(params, isa, threads, "mxfp4",
                                             (pr.x.shape[0], w.n, w.k))))
    _check(rc, "lb_gemm_mxfp4")
    return pr.y


def gemm_w4a8(x, w: Int4Weights, params: Params | None = None, isa=None,
              threads: int | None = None, _prep: _Prepared | None = None) -> np.ndarray:
    """fp32 X in; X is quantized per token to int8 inside the call (counted in timings)."""
    pr = _prep or _Prepared(x, w.kp, w.n)
    rc = lib().lb_gemm_w4a8(pr.x, pr.x.shape[0], w.kp, w.qw, w.scales, _zp(w.zeros), w.n,
                            pr.y, C.byref(_params(params, isa, threads, "w4a8",
                                                   (pr.x.shape[0], w.n, w.k))))
    _check(rc, "lb_gemm_w4a8")
    return pr.y


def gemm_w4a8_q(a: Int8Acts, w: Int4Weights, params: Params | None = None, isa=None,
                threads: int | None = None) -> np.ndarray:
    y = np.empty((a.xq.shape[0], w.n), np.float32)
    rc = lib().lb_gemm_w4a8_q(a.xq, a.scale, a.gsum, a.xq.shape[0], w.kp, w.qw, w.scales,
                              _zp(w.zeros), w.n, y,
                              C.byref(_params(params, isa, threads, "w4a8", (a.xq.shape[0], w.n, w.k))))
    _check(rc, "lb_gemm_w4a8_q")
    return y


def quant_act_int8(x: np.ndarray, kp: int) -> Int8Acts:
    xp = pad_acts(np.atleast_2d(x), kp)
    m = xp.shape[0]
    xq = np.empty((m, kp), np.int8)
    xs = np.empty(m, np.float32)
    gs = np.empty((m, kp // GROUP), np.int32)
    lib().lb_quant_act_int8(xp, m, kp, xq, xs, gs)
    return Int8Acts(xq, xs, gs)


def w4a8_group_acc(a: Int8Acts, w: Int4Weights, isa=None) -> np.ndarray:
    m = a.xq.shape[0]
    out = np.empty((m, w.n, w.kp // GROUP), np.int32)
    _check(lib().lb_w4a8_group_acc(a.xq, a.gsum, m, w.kp, w.qw, _zp(w.zeros), w.n, out,
                                   _isa_id(isa)), "lb_w4a8_group_acc")
    return out


def gemm_w4a8_amx(x, w: AMXInt4Weights, params=None, isa=None, threads: int | None = None,
                  _prep: _Prepared | None = None) -> np.ndarray:
    """W4A8 on AMX tiles (fp32 X in, int8 per-token quantization inside)."""
    if not amx_available():
        raise RuntimeError("AMX-INT8 is not available on this CPU/OS")
    pr = _prep or _Prepared(x, w.kp, w.n)
    rc = lib().lb_gemm_w4a8_amx(pr.x, pr.x.shape[0], w.kp, w.qw, w.scales, w.sz, w.n, pr.y,
                                threads or 0)
    _check(rc, "lb_gemm_w4a8_amx")
    return pr.y


def w4a8_group_acc_amx(a: Int8Acts, w: AMXInt4Weights) -> np.ndarray:
    m = a.xq.shape[0]
    out = np.empty((m, w.n, w.kp // GROUP), np.int32)
    _check(lib().lb_w4a8_group_acc_amx(a.xq, a.gsum, m, w.kp, w.qw, w.zeros, w.n, out),
           "lb_w4a8_group_acc_amx")
    return out


GEMMS = {"w4a16": gemm_w4a16, "mxfp4": gemm_mxfp4, "w4a8": gemm_w4a8}
