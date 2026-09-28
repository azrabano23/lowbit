"""`lowbit` command line: info, build, tune, bench, report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _shape(s: str) -> tuple[int, int, int]:
    parts = [int(v) for v in s.lower().replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be MxNxK")
    return tuple(parts)  # type: ignore[return-value]


def cmd_info(a) -> int:
    from . import kernels as K
    from ._build import build
    print(json.dumps({"library": str(build()), **K.cpu_info(),
                      "effective": {k: K.effective_isa(k) for k in K.KIND_IDS}}, indent=1))
    return 0


def cmd_build(a) -> int:
    from ._build import build
    print(build(verbose=True, force=a.force))
    return 0


def cmd_tune(a) -> int:
    from . import tune as T
    path = Path(a.cache) if a.cache else T.DEFAULT_CACHE
    for shape in a.shape:
        for kind in a.kernel:
            e = T.tune(kind, *shape, isa=a.isa, threads=a.threads, reps=a.reps, path=path,
                       verbose=a.verbose)
            print(f"{kind} {'x'.join(map(str, shape))}: {e['params']}  "
                  f"{e['gflops']:.1f} GFLOP/s  ({e['default_time_s'] / e['time_s']:.2f}x vs "
                  f"default, {e['n_configs']} configs)")
    print(f"cache: {path}")
    return 0


def cmd_bench(a) -> int:
    from . import bench as B
    from . import report as R
    out = Path(a.out)
    if a.quick:
        shapes = B.QUICK_DECODE + B.QUICK_BATCH
    else:
        shapes = a.shape or (B.DECODE_SHAPES + B.BATCH_SHAPES)
    tc = out / "tune_cache.json"
    res = B.run(shapes=shapes, threads_list=a.threads, isas=a.isa, min_time=a.min_time,
                tune_cache=tc, do_tune=not a.no_tune, cold=not a.no_cold,
                idle_threshold=a.wait_idle)
    p = B.save(res, out)
    print(f"wrote {p}")
    for w in R.write(out, a.readme):
        print(f"wrote {w}")
    return 0


def cmd_report(a) -> int:
    from . import report as R
    for w in R.write(a.results, a.readme):
        print(f"wrote {w}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lowbit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="show detected CPU features and the ISA each kernel uses")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("build", help="compile the C kernels (normally done lazily)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("tune", help="autotune tile sizes for shapes and cache them")
    s.add_argument("--shape", type=_shape, action="append", required=True, help="MxNxK")
    s.add_argument("--kernel", choices=["w4a16", "w4a8", "mxfp4"], action="append")
    s.add_argument("--isa", choices=["scalar", "avx2", "avx512"])
    s.add_argument("--threads", type=int)
    s.add_argument("--reps", type=int, default=5)
    s.add_argument("--cache", help="JSON cache path (default ~/.cache/lowbit/tune.json)")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("bench", help="run benchmarks, write results/bench.json + RESULTS.md")
    s.add_argument("--out", default="results")
    s.add_argument("--shape", type=_shape, action="append")
    s.add_argument("--threads", type=int, action="append")
    s.add_argument("--isa", choices=["scalar", "avx2", "avx512"], action="append")
    s.add_argument("--min-time", type=float, default=0.4)
    s.add_argument("--quick", action="store_true", help="two small shapes only")
    s.add_argument("--no-tune", action="store_true")
    s.add_argument("--no-cold", action="store_true", help="skip the cold-cache decode runs")
    s.add_argument("--readme", help="also refresh the results block in this README")
    s.add_argument("--wait-idle", type=float, default=0.0, metavar="FRAC",
                   help="before each block wait until this fraction of CPU is idle "
                        "(e.g. 0.9 on a shared box)")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("report", help="regenerate markdown from results/bench.json")
    s.add_argument("--results", default="results")
    s.add_argument("--readme")
    s.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    if a.cmd == "tune" and not a.kernel:
        a.kernel = ["w4a16", "w4a8", "mxfp4"]
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
