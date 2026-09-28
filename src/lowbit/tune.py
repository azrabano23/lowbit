"""Tile-size autotuner with a JSON cache.

The search space per (kernel, M, N, K, isa, threads):
  micro-tile (mr x nr) from kernels.TILES, N-tile nb, K-block kb, loop order.
A full grid is ~500 configs, so we search in two stages:
  1. all micro-tiles with a shape-derived nb, kb = whole K, order 0;
  2. for the `top` (default 3) best micro-tiles, the grid nb x kb x order.
Each candidate is timed as the minimum of a few runs after a warm-up.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import kernels as K
from .formats import quantize_int4, quantize_mxfp4

DEFAULT_CACHE = Path(os.environ.get("LOWBIT_TUNE_CACHE",
                                    Path.home() / ".cache" / "lowbit" / "tune.json"))


def cpu_key() -> str:
    model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model}|{os.cpu_count()}c"


def key(kind: str, m: int, n: int, k: int, isa: str, threads: int) -> str:
    return f"{kind}|{isa}|t{threads}|{m}x{n}x{k}"


def load_cache(path: Path | str = DEFAULT_CACHE) -> dict:
    p = Path(path)
    if not p.exists():
        return {"cpu": cpu_key(), "entries": {}}
    data = json.loads(p.read_text())
    if data.get("cpu") != cpu_key():  # tuned on another machine: ignore
        return {"cpu": cpu_key(), "entries": {}}
    return data


def save_cache(data: dict, path: Path | str = DEFAULT_CACHE) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, p)


def lookup(kind: str, m: int, n: int, k: int, isa: str | None = None,
           threads: int | None = None, path: Path | str = DEFAULT_CACHE) -> K.Params | None:
    isa_e = K.effective_isa(kind, isa)
    th = threads or K.cpu_info()["max_threads"]
    e = load_cache(path)["entries"].get(key(kind, m, n, k, isa_e, th))
    return K.Params(**e["params"]) if e else None


def _time(fn, reps: int) -> float:
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _problem(kind: str, m: int, n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    x = rng.standard_normal((m, k)).astype(np.float32)
    wq = quantize_mxfp4(w) if kind == "mxfp4" else quantize_int4(w)
    return x, wq


def candidates_stage1(m: int, n: int, threads: int, isa: str):
    if isa == "scalar":
        return [(1, 1)]
    return [(a, b) for a, b in K.TILES if a <= max(1, m) and b <= n]


def default_nb(n: int, threads: int) -> int:
    # enough tiles for every thread, rounded down to a power of two in [8, 256]
    nb = max(8, min(256, n // max(1, 4 * threads)))
    return 1 << (nb.bit_length() - 1)


def tune(kind: str, m: int, n: int, k: int, isa: str | None = None,
         threads: int | None = None, reps: int = 5, problem=None, top: int = 3,
         path: Path | str | None = DEFAULT_CACHE, verbose: bool = False) -> dict:
    """Search tile parameters; returns the cache entry and stores it if path is set."""
    isa_e = K.effective_isa(kind, isa)
    th = threads or K.cpu_info()["max_threads"]
    x, wq = problem if problem is not None else _problem(kind, m, n, k)
    gemm = K.GEMMS[kind]
    prep = K._Prepared(x, wq.kp, wq.n)
    tried: dict[K.Params, float] = {}

    def run(p: K.Params) -> float:
        if p not in tried:
            tried[p] = _time(lambda: gemm(x, wq, p, isa=isa_e, threads=th, _prep=prep), reps)
            if verbose:
                print(f"  {p}  {tried[p] * 1e3:.3f} ms")
        return tried[p]

    nb0 = default_nb(n, th)
    for mr, nr in candidates_stage1(m, n, th, isa_e):
        run(K.Params(mr=mr, nr=nr, nb=max(nb0, nr), kb=0, order=0))
    top_cfgs = sorted(tried, key=tried.get)[:top]
    kp = wq.kp
    kbs = [0] + [b for b in (1024, 2048) if b < kp]
    nbs = sorted({b for b in (16, 32, 64, 128, 256) if b <= max(16, n)} | {nb0})
    orders = [0, 1] if m > 1 else [0]
    for base in top_cfgs:
        for nb in nbs:
            if nb < base.nr:
                continue
            for kb in kbs:
                for order in orders:
                    run(K.Params(mr=base.mr, nr=base.nr, nb=nb, kb=kb, order=order))
    best = min(tried, key=tried.get)
    default_t = run(K.Params())
    entry = {
        "params": best.as_dict(),
        "time_s": tried[best],
        "default_time_s": default_t,
        "n_configs": len(tried),
        "gflops": 2.0 * m * n * k / tried[best] / 1e9,
    }
    if path is not None:
        data = load_cache(path)
        data["entries"][key(kind, m, n, k, isa_e, th)] = entry
        save_cache(data, path)
    return entry
