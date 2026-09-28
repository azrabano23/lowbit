"""Forcing the non-AVX-512 paths with LOWBIT_ISA (what a CI runner without AVX-512 runs)."""
import os
import subprocess
import sys

import pytest
from conftest import ISAS

SCRIPT = r"""
import numpy as np, lowbit as lb
from lowbit import kernels as K, reference as R
info = K.cpu_info()
print(info["active_isa"])
rng = np.random.default_rng(0)
for m, n, k in [(1, 37, 300), (9, 64, 1024)]:
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k)).astype(np.float32)
    W, Wm = lb.quantize_int4(w, asym=True), lb.quantize_mxfp4(w)
    for kind, got, ref, wd in [
        ("w4a16", K.gemm_w4a16(x, W, K.Params(mr=4, nr=2)), R.gemm_w4a16_ref(x, W), W),
        ("mxfp4", K.gemm_mxfp4(x, Wm, K.Params(mr=2, nr=4)), R.gemm_mxfp4_ref(x, Wm), Wm),
    ]:
        assert K.effective_isa(kind) == info["active_isa"], kind
        bound = np.abs(x) @ np.abs(wd.dequantize()).T
        assert np.all(np.abs(got - ref) <= 1e-5 * bound), kind
    a = lb.quantize_act_int8(x)
    assert (K.w4a8_group_acc(a, W) == R.w4a8_group_acc_ref(a, W)).all()
    y = K.gemm_w4a8(x, W)
    assert np.allclose(y, R.gemm_w4a8_ref(x, W), rtol=1e-4, atol=1e-3)
print("OK")
"""


@pytest.mark.parametrize("forced", ["scalar", "avx2"])
def test_forced_isa(forced):
    if forced not in ISAS:
        pytest.skip(f"CPU cannot run {forced}")
    env = dict(os.environ, LOWBIT_ISA=forced)
    r = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    active, ok = r.stdout.split()
    assert active == forced and ok == "OK"


def test_env_cap_never_raises_isa():
    env = dict(os.environ, LOWBIT_ISA="avx512")
    r = subprocess.run([sys.executable, "-c", "from lowbit import kernels as K; "
                        "print(K.cpu_info()['active_isa'])"], env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    # a cap above the hardware is a no-op: we get exactly what the CPU supports
    from lowbit import kernels as K
    assert r.stdout.strip() == K.cpu_info()["cpu_isa"]
