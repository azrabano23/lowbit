"""Benchmarks: fused low-bit kernels vs numpy (OpenBLAS) baselines.

Methods (all compute Y = X @ W^T for the same quantized W):
  numpy fp32          x @ W_fp32.T with W dequantized ahead of time (what you do
                      without a fused kernel: 8x the weight bytes of int4)
  numpy deq+mm        dequantize the packed weights with numpy, then matmul
                      (dequant time included) - the naive "load then convert"
  <fmt> scalar        this library's portable C path (no SIMD, -fno-tree-vectorize)
  <fmt> avx2 / avx512 this library's SIMD paths with autotuned tile sizes
  w4a8 amx            AMX-INT8 tile path on a one-time repacked weight layout

Cache modes for decode (M=1):
  hot   the same weights every call (they may sit in the LLC)
  cold  rotate through enough copies of the weights to exceed ~2x the LLC, so
        every call streams weights from DRAM like real LLM decode does
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import kernels as K
from . import tune as T
from .formats import pack_amx, quantize_int4, quantize_mxfp4

DECODE_SHAPES = [(1, 4096, 4096), (1, 11008, 4096), (1, 4096, 11008)]
BATCH_SHAPES = [(16, 4096, 4096), (64, 4096, 4096), (16, 11008, 4096), (64, 11008, 4096)]
QUICK_DECODE = [(1, 4096, 4096)]
QUICK_BATCH = [(16, 4096, 4096)]
FORMATS = ["w4a16", "w4a8", "mxfp4"]


def llc_bytes() -> int:
    best = 0
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    for idx in base.glob("index*"):
        try:
            txt = (idx / "size").read_text().strip()
        except OSError:
            continue
        mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}.get(txt[-1], 1)
        val = int(txt[:-1]) if txt[-1] in "KMG" else int(txt)
        best = max(best, val * mult)
    return best or 32 << 20


def cpu_idle_fraction(interval: float = 0.5) -> float:
    """Fraction of all CPUs idle over `interval` (Linux /proc/stat; 1.0 elsewhere)."""
    def snap():
        f = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        v = [int(t) for t in f]
        return v[3] + v[4], sum(v)  # idle + iowait, total
    try:
        i0, t0 = snap()
        time.sleep(interval)
        i1, t1 = snap()
        return (i1 - i0) / max(1, t1 - t0)
    except (OSError, IndexError, ValueError):
        return 1.0


def wait_idle(threshold: float, timeout: float, log=print) -> float:
    """Block until the machine is at least `threshold` idle (other tenants quiet)."""
    t_end = time.time() + timeout
    idle = cpu_idle_fraction()
    warned = False
    while idle < threshold and time.time() < t_end:
        if not warned:
            log(f"  waiting for an idle machine (idle {idle:.0%} < {threshold:.0%})")
            warned = True
        time.sleep(2.0)
        idle = cpu_idle_fraction()
    return idle


def _busy_s() -> float:
    """CPU-seconds all tenants have spent non-idle (Linux /proc/stat)."""
    v = [int(t) for t in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return (sum(v) - v[3] - v[4]) / os.sysconf("SC_CLK_TCK")


def _self_s() -> float:
    t = os.times()
    return t.user + t.system  # all threads of this process


def measure_contended(fn, min_time: float, max_other: float, retries: int,
                      idle_timeout: float, log=print):
    """Time fn; if other processes used more than `max_other` cores meanwhile,
    wait for quiet and retry. Returns (times, other_cores, attempts)."""
    best = None
    for attempt in range(1, retries + 1):
        try:
            b0, s0, w0 = _busy_s(), _self_s(), time.perf_counter()
            ts = _timeit(fn, min_time)
            b1, s1, w1 = _busy_s(), _self_s(), time.perf_counter()
            other = max(0.0, ((b1 - b0) - (s1 - s0)) / max(1e-9, w1 - w0))
        except (OSError, ValueError, IndexError):
            return _timeit(fn, min_time), None, 1
        if best is None or other < best[1]:
            best = (ts, other, attempt)
        if other <= max_other:
            break
        log(f"    other tenants used {other:.2f} cores during the run; retrying")
        wait_idle(0.9, idle_timeout, log)
    return best


def _timeit(fn, min_time: float, min_reps: int = 3, max_reps: int = 200) -> list[float]:
    fn()  # warm-up (also builds/loads, spins up thread pools)
    ts: list[float] = []
    t_end = time.perf_counter() + min_time
    while len(ts) < max_reps and (len(ts) < min_reps or time.perf_counter() < t_end):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def _blas_threads(n: int):
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n, user_api="blas")
    except ImportError:  # pragma: no cover
        import contextlib
        return contextlib.nullcontext()


def _rel_err(y, ref) -> float:
    return float(np.max(np.abs(y - ref)) / (np.max(np.abs(ref)) + 1e-30))


def run(shapes=None, threads_list=None, isas=None, min_time: float = 0.4,
        tune_cache: Path | str | None = None, do_tune: bool = True, cold: bool = True,
        seed: int = 0, idle_threshold: float = 0.0, idle_timeout: float = 600.0,
        max_other: float = 0.25, retries: int = 4, log=print) -> dict:
    info = K.cpu_info()
    shapes = shapes or DECODE_SHAPES + BATCH_SHAPES
    threads_list = threads_list or sorted({1, info["max_threads"]})
    if isas is None:
        order = ["scalar", "avx2", "avx512"]
        isas = order[: order.index(info["active_isa"]) + 1]
    llc = llc_bytes()
    rows = []
    t_start = time.time()
    load0 = os.getloadavg()
    rng = np.random.default_rng(seed)
    for (m, n, k) in shapes:
        log(f"== shape M={m} N={n} K={k}")
        w = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
        x = rng.standard_normal((m, k)).astype(np.float32)
        wq = {"w4a16": quantize_int4(w), "mxfp4": quantize_mxfp4(w)}
        wq["w4a8"] = wq["w4a16"]
        w_fp32 = wq["w4a16"].dequantize()  # the pre-dequantized baseline weights
        del w
        ref = {f: x.astype(np.float64) @ wq[f].dequantize().T.astype(np.float64) for f in ("w4a16", "mxfp4")}
        ref["w4a8"] = ref["w4a16"]
        phase = "decode" if m == 1 else "batch"
        flops = 2.0 * m * n * k
        modes = ["hot", "cold"] if (cold and m == 1) else ["hot"]

        def pool(obj, nbytes):
            copies = max(1, int(np.ceil(max(2 * llc, 512 << 20) / nbytes)))
            copies = min(copies, max(1, (3 << 30) // nbytes))
            if copies == 1:
                return [obj]
            if isinstance(obj, np.ndarray):
                return [obj] + [obj.copy() for _ in range(copies - 1)]
            owned = [f for f in ("qw", "scales", "e8m0", "sz") if hasattr(obj, f)]
            return [obj] + [replace(obj, **{f: getattr(obj, f).copy() for f in owned})
                            for _ in range(copies - 1)]

        amx, amx_pool = None, {}
        if "avx512" in isas and K.amx_available():
            t0 = time.perf_counter()
            amx = pack_amx(wq["w4a8"])  # one-time repack, like loading a model
            log(f"  AMX repack (numpy, one-time): {time.perf_counter() - t0:.2f} s")
            amx_pool = {"hot": [amx]}
            if "cold" in modes:
                amx_pool["cold"] = pool(amx, amx.nbytes)

        for mode in modes:
            wf_pool = pool(w_fp32, w_fp32.nbytes) if mode == "cold" else [w_fp32]
            wq_pool = {f: (pool(wq[f], wq[f].nbytes) if mode == "cold" else [wq[f]])
                       for f in ("w4a16", "mxfp4")}
            wq_pool["w4a8"] = wq_pool["w4a16"]
            for th in threads_list:
                base_t = None
                idle = wait_idle(idle_threshold, idle_timeout, log) if idle_threshold else \
                    cpu_idle_fraction(0.2)

                def record(method, fmt, isa, fn, wbytes, params=None, check=None):
                    nonlocal base_t
                    if idle_threshold:
                        ts, other, att = measure_contended(fn, min_time, max_other, retries,
                                                           idle_timeout, log)
                    else:
                        ts, other, att = _timeit(fn, min_time), None, 1
                    med, mn = float(np.median(ts)), float(np.min(ts))
                    if method == "numpy fp32":
                        base_t = med
                    row = {
                        "phase": phase, "M": m, "N": n, "K": k, "threads": th, "cache": mode,
                        "method": method, "format": fmt, "isa": isa, "params": params,
                        "reps": len(ts), "t_median_s": med, "t_min_s": mn,
                        "gflops": flops / med / 1e9, "weight_bytes": int(wbytes),
                        "weight_gbps": wbytes / med / 1e9,
                        "speedup_vs_numpy_fp32": (base_t / med) if base_t else None,
                        "max_rel_err": check, "idle_before": round(idle, 3),
                        "other_tenant_cores": None if other is None else round(other, 3),
                        "attempts": att,
                    }
                    rows.append(row)
                    log(f"  [{mode} t={th}] {method:<22} {med * 1e3:9.3f} ms  "
                        f"{row['gflops']:7.1f} GFLOP/s  {row['weight_gbps']:6.1f} GB/s  "
                        f"x{row['speedup_vs_numpy_fp32'] or 1:.2f}")

                it = {"i": 0}

                def nxt(p):
                    it["i"] = (it["i"] + 1) % len(p)
                    return p[it["i"]]

                with _blas_threads(th):
                    y = x @ w_fp32.T
                    record("numpy fp32", "fp32", "blas", lambda: x @ nxt(wf_pool).T,
                           w_fp32.nbytes, check=_rel_err(y, ref["w4a16"]))
                    if mode == "hot":
                        for f in ("w4a16", "mxfp4"):
                            y = x @ wq[f].dequantize().T
                            record(f"numpy deq+mm {f}", f, "numpy",
                                   lambda f=f: x @ wq[f].dequantize().T, wq[f].nbytes,
                                   check=_rel_err(y, ref[f]))
                for fmt in FORMATS:
                    gemm = K.GEMMS[fmt]
                    for isa in isas:
                        if mode == "cold" and isa == "scalar":
                            continue  # compute bound; cold changes nothing
                        eff = K.effective_isa(fmt, isa)
                        if eff != isa:
                            continue
                        if isa == "scalar":
                            params = K.Params()
                        else:
                            params = T.lookup(fmt, m, n, k, isa, th, path=tune_cache) \
                                if tune_cache else None
                            if params is None and do_tune:
                                if idle_threshold:
                                    wait_idle(idle_threshold, idle_timeout, log)
                                e = T.tune(fmt, m, n, k, isa=isa, threads=th, reps=3,
                                           problem=(x, wq[fmt]), top=3 if isa == isas[-1] else 1,
                                           path=tune_cache)
                                params = K.Params(**e["params"])
                                log(f"  tuned {fmt}/{isa}/t{th}: {e['params']} "
                                    f"({e['n_configs']} configs)")
                            params = params or K.Params()
                        prep = K._Prepared(x, wq[fmt].kp, n)
                        y = gemm(x, wq[fmt], params, isa=isa, threads=th).copy()
                        err = _rel_err(y, ref[fmt])
                        pl = wq_pool[fmt]
                        label = {"w4a8": {"avx512": "avx512-vnni", "avx2": "avx2-maddubs"}}.get(
                            fmt, {}).get(isa, isa)
                        record(f"{fmt} {label}", fmt, isa,
                               lambda: gemm(x, nxt(pl), params, isa=isa, threads=th, _prep=prep),
                               wq[fmt].nbytes, params=params.as_dict(), check=err)
                    if fmt == "w4a8" and amx is not None:
                        prep = K._Prepared(x, amx.kp, n)
                        err = _rel_err(K.gemm_w4a8_amx(x, amx, threads=th).copy(), ref[fmt])
                        pl_amx = amx_pool[mode]
                        record("w4a8 amx", "w4a8", "amx",
                               lambda: K.gemm_w4a8_amx(x, nxt(pl_amx), threads=th, _prep=prep),
                               amx.nbytes, check=err)
    meta = {
        "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "cpu": T.cpu_key(), "lowbit": info, "llc_bytes": llc,
        "numpy": np.__version__, "blas": _blas_info(), "python": platform.python_version(),
        "loadavg_start": load0, "loadavg_end": os.getloadavg(),
        "wall_s": round(time.time() - t_start, 1), "min_time_per_method_s": min_time,
        "timing": "median of >=3 reps after 1 warm-up (t_min_s also recorded)",
        "contention_control": (f"wait for >= {idle_threshold:.0%} idle CPU before each block; "
                               f"re-measure (up to {retries}x) when other processes used "
                               f"> {max_other} cores during a measurement") if idle_threshold
        else "none",
    }
    return {"meta": meta, "rows": rows}


def _blas_info() -> str:
    try:
        from threadpoolctl import threadpool_info
        for d in threadpool_info():
            if d.get("user_api") == "blas":
                return f"{d.get('internal_api')} {d.get('version')} ({d.get('architecture')})"
    except ImportError:  # pragma: no cover
        pass
    return "unknown"


def save(result: dict, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "bench.json"
    p.write_text(json.dumps(result, indent=1))
    return p
