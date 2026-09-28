import numpy as np
import pytest

from lowbit import kernels as K

ISA_ORDER = ["scalar", "avx2", "avx512"]


def available_isas() -> list[str]:
    top = ISA_ORDER.index(K.cpu_info()["active_isa"])
    return ISA_ORDER[: top + 1]


ISAS = available_isas()


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def assert_fp_close(y, ref, bound, rtol=1e-5):
    """|y - ref| <= rtol * (|x| @ |W|^T): the natural scale of fp32 dot-product error."""
    err = np.abs(np.asarray(y, np.float64) - ref)
    lim = rtol * bound + 1e-30
    worst = np.max(err / lim)
    assert worst <= 1.0, f"max err/limit = {worst:.3g} (max abs err {err.max():.3g})"
