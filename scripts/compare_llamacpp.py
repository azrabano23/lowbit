"""Head-to-head against llama.cpp's ggml CPU backend on the same GEMM shapes.

Runs ggml's `test-backend-ops perf -o MUL_MAT` (m=4096, k=14336, n in {1, 8};
ggml's m/n are our N/M) for f32, q4_0 and mxfp4 weights, then this library's
kernels on the identical shapes, and writes results/llamacpp.json.

How llama.cpp was built (see README "What is real"):
  git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
  # one-line patch so the perf mode honours a thread count (default is ncpu/2):
  #   tests/test-backend-ops.cpp: ggml_backend_set_n_threads_fn(..., getenv("GGML_BENCH_THREADS") ? ...)
  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON
  ninja -C build test-backend-ops

Usage: python scripts/compare_llamacpp.py --bin /path/to/test-backend-ops [--out results]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

import lowbit as lb
from lowbit import bench as B
from lowbit import kernels as K
from lowbit import tune as T
from lowbit.formats import pack_amx

N, KD = 4096, 14336
MS = [1, 8]
LINE = re.compile(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+).*?"
                  r"(\d+) runs -\s+([\d.]+) us/run.*?([\d.]+) (G|T|M)FLOPS")


def run_ggml(binary: str, threads: int) -> tuple[list[dict], str]:
    env = dict(os.environ, GGML_BENCH_THREADS=str(threads))
    pat = f"type_a=(q4_0|mxfp4|f32),type_b=f32,m={N},n=({'|'.join(map(str, MS))}),k={KD},"
    r = subprocess.run([binary, "perf", "-b", "CPU", "-o", "MUL_MAT", "-p", pat], env=env,
                       capture_output=True, text=True, timeout=1800)
    raw = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
    rows = []
    for m in LINE.finditer(raw):
        t, _, n, _, runs, us, _, _ = m.groups()
        rows.append({"impl": "llama.cpp/ggml", "type": t, "M": int(n), "N": N, "K": KD,
                     "threads": threads, "runs": int(runs), "t_s": float(us) * 1e-6,
                     "gflops": 2.0 * int(n) * N * KD / (float(us) * 1e-6) / 1e9})
    return rows, raw


def run_lowbit(threads: int, tune_cache: Path, min_time: float, log=print) -> list[dict]:
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((N, KD)) * 0.02).astype(np.float32)
    w4 = lb.quantize_int4(w)
    mx = lb.quantize_mxfp4(w)
    amx = pack_amx(w4) if K.amx_available() else None
    rows = []
    for m in MS:
        x = rng.standard_normal((m, KD)).astype(np.float32)
        cases = [("w4a16", w4, K.gemm_w4a16), ("w4a8", w4, K.gemm_w4a8), ("mxfp4", mx, K.gemm_mxfp4)]
        for kind, wq, fn in cases:
            p = T.lookup(kind, m, N, KD, None, threads, path=tune_cache)
            if p is None:
                B.wait_idle(0.9, 600, log)
                p = K.Params(**T.tune(kind, m, N, KD, threads=threads, reps=3, problem=(x, wq),
                                      path=tune_cache)["params"])
            prep = K._Prepared(x, wq.kp, N)
            ts, other, _ = B.measure_contended(lambda: fn(x, wq, p, threads=threads, _prep=prep),
                                               min_time, 0.25, 4, 600, log)
            t = float(np.median(ts))
            rows.append({"impl": "lowbit", "type": kind + (" avx512-vnni" if kind == "w4a8" else ""),
                         "M": m, "N": N, "K": KD, "threads": threads, "runs": len(ts), "t_s": t,
                         "gflops": 2.0 * m * N * KD / t / 1e9, "params": p.as_dict(),
                         "other_tenant_cores": other})
        if amx is not None:
            prep = K._Prepared(x, amx.kp, N)
            ts, other, _ = B.measure_contended(
                lambda: K.gemm_w4a8_amx(x, amx, threads=threads, _prep=prep), min_time, 0.25, 4,
                600, log)
            t = float(np.median(ts))
            rows.append({"impl": "lowbit", "type": "w4a8 amx", "M": m, "N": N, "K": KD,
                         "threads": threads, "runs": len(ts), "t_s": t,
                         "gflops": 2.0 * m * N * KD / t / 1e9, "other_tenant_cores": other})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to llama.cpp build/bin/test-backend-ops")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, action="append")
    ap.add_argument("--min-time", type=float, default=0.5)
    ap.add_argument("--commit", default="", help="llama.cpp commit hash (recorded)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    threads = a.threads or sorted({1, K.cpu_info()["max_threads"]})
    rows, raws = [], []
    for th in threads:
        B.wait_idle(0.9, 600)
        r, raw = run_ggml(a.bin, th)
        print(raw.strip().splitlines()[-4:] if raw.strip() else "no ggml output")
        rows += r
        raws.append(f"### threads={th}\n{raw}")
        rows += run_lowbit(th, out / "tune_cache.json", a.min_time)
    (out / "llamacpp_raw.txt").write_text("\n".join(raws))
    res = {"meta": {"date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "cpu": T.cpu_key(), "llama_cpp_commit": a.commit,
                    "ggml_bench": "test-backend-ops perf -o MUL_MAT (CPU backend, default buffer "
                                  "type: no Q4_0 repacking / AMX extra buffers)",
                    "note": "ggml m/n = our N/M. ggml times are its own mean us/run; ours are "
                            "medians. Both reuse the same weights every call (hot cache)."},
           "rows": rows}
    (out / "llamacpp.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"{r['impl']:<15} {r['type']:<18} M={r['M']:<2} t={r['threads']} "
              f"{r['t_s'] * 1e3:8.3f} ms {r['gflops']:7.1f} GFLOP/s")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f}s")
