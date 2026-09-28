"""Lazy build of the C kernels into a shared library (no build deps beyond a C compiler).

The library is compiled on first use and cached under a name that hashes the
sources and flags, so editing a .c file triggers a rebuild automatically.

Environment:
  CC                 compiler (default: cc)
  LOWBIT_LIB         path to a prebuilt liblowbit shared library (skips the build)
  LOWBIT_BUILD_DIR   where to put the built library (default: <package>/_build,
                     falling back to ~/.cache/lowbit if that is not writable)
  LOWBIT_NO_OPENMP=1 build without OpenMP
  LOWBIT_CFLAGS      extra compiler flags (e.g. "-DLB_PF=512" prefetch distance)
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CSRC = Path(__file__).parent / "csrc"
BASE = ["-O3", "-std=c11", "-fPIC", "-Wall", "-Wextra", "-Wno-unused-parameter"]
IS_X86 = platform.machine().lower() in ("x86_64", "amd64", "i686", "x86")

# (source, extra flags). Each ISA's file gets only its own -m flags; the
# dispatcher checks CPUID before calling into it.
UNITS = [
    ("dispatch.c", []),
    ("kern_scalar.c", ["-O2", "-fno-tree-vectorize"]),
]
if IS_X86:
    UNITS += [
        ("kern_avx2.c", ["-mavx2", "-mfma"]),
        ("kern_avx512.c", ["-mavx2", "-mfma", "-mavx512f", "-mavx512bw"]),
        ("kern_vnni.c", ["-mavx2", "-mfma", "-mavx512f", "-mavx512bw", "-mavx512vnni"]),
    ]
# optional units: skipped (with a warning) if the compiler cannot build them
OPTIONAL = {"kern_amx.c"}
if IS_X86 and sys.platform.startswith("linux"):
    UNITS += [("kern_amx.c", ["-mavx2", "-mfma", "-mavx512f", "-mavx512bw", "-mamx-tile",
                              "-mamx-int8"])]


def _ext() -> str:
    return {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")


def _cc() -> str:
    return os.environ.get("CC", "cc")


def _openmp_ok(cc: str) -> bool:
    if os.environ.get("LOWBIT_NO_OPENMP"):
        return False
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "t.c"
        src.write_text("#include <omp.h>\nint main(void){return omp_get_max_threads()>0?0:1;}\n")
        r = subprocess.run([cc, "-fopenmp", str(src), "-o", str(Path(d) / "t")],
                           capture_output=True)
        return r.returncode == 0


def _digest(cc: str, omp: bool) -> str:
    h = hashlib.sha256()
    for p in sorted(CSRC.iterdir()):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(repr((cc, omp, BASE, UNITS, _extra())).encode())
    return h.hexdigest()[:12]


def _extra() -> list[str]:
    return os.environ.get("LOWBIT_CFLAGS", "").split()


def _build_dir() -> Path:
    env = os.environ.get("LOWBIT_BUILD_DIR")
    cands = [Path(env)] if env else []
    cands += [Path(__file__).parent / "_build", Path.home() / ".cache" / "lowbit"]
    for d in cands:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("")
            probe.unlink()
            return d
        except OSError:
            continue
    raise RuntimeError("no writable build directory for lowbit")


def build(verbose: bool = False, force: bool = False) -> Path:
    """Compile (if needed) and return the path of the shared library."""
    if os.environ.get("LOWBIT_LIB"):
        return Path(os.environ["LOWBIT_LIB"])
    cc = _cc()
    if shutil.which(cc) is None:
        raise RuntimeError(f"C compiler {cc!r} not found; set CC or LOWBIT_LIB")
    omp = _openmp_ok(cc)
    out_dir = _build_dir()
    lib = out_dir / f"liblowbit-{_digest(cc, omp)}{_ext()}"
    if lib.exists() and not force:
        return lib
    defs = ["-DLB_X86"] if IS_X86 else []
    ompf = ["-fopenmp"] if omp else []
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        objs = []
        for src, flags in UNITS:
            obj = Path(tmp) / (src[:-2] + ".o")
            cmd = [cc, *BASE, *flags, *defs, *ompf, *_extra(), "-I", str(CSRC), "-c",
                   str(CSRC / src), "-o", str(obj)]
            if verbose:
                print(" ".join(cmd))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                if src in OPTIONAL:
                    print(f"lowbit: skipping optional {src} (compiler lacks support)",
                          file=sys.stderr)
                    continue
                raise RuntimeError(f"compiling {src} failed:\n{r.stderr}")
            if r.stderr.strip() and verbose:
                print(r.stderr, file=sys.stderr)
            objs.append(str(obj))
        tmp_lib = Path(tmp) / lib.name
        cmd = [cc, "-shared", *ompf, *objs, "-o", str(tmp_lib), "-lm"]
        if verbose:
            print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        os.replace(tmp_lib, lib)  # atomic: concurrent test workers are safe
    for old in out_dir.glob(f"liblowbit-*{_ext()}"):
        if old != lib:
            try:
                old.unlink()
            except OSError:
                pass
    return lib


if __name__ == "__main__":
    print(build(verbose=True, force="--force" in sys.argv))
