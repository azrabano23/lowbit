import json

import numpy as np

import lowbit as lb
from lowbit import kernels as K
from lowbit import reference as R
from lowbit import tune


def test_tune_caches_and_result_is_correct(tmp_path):
    cache = tmp_path / "tune.json"
    e = tune.tune("w4a16", 3, 48, 256, threads=1, reps=1, path=cache)
    assert e["n_configs"] > 1 and e["time_s"] <= e["default_time_s"] * 1.5
    data = json.loads(cache.read_text())
    assert data["cpu"] == tune.cpu_key() and len(data["entries"]) == 1
    p = tune.lookup("w4a16", 3, 48, 256, threads=1, path=cache)
    assert p == K.Params(**e["params"])
    assert tune.lookup("w4a16", 3, 48, 512, threads=1, path=cache) is None
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 256)).astype(np.float32)
    W = lb.quantize_int4(rng.standard_normal((48, 256)).astype(np.float32))
    y = K.gemm_w4a16(x, W, p, threads=1)
    np.testing.assert_allclose(y, R.gemm_w4a16_ref(x, W), rtol=1e-4, atol=1e-4)


def test_cache_from_other_cpu_is_ignored(tmp_path):
    cache = tmp_path / "tune.json"
    cache.write_text(json.dumps({"cpu": "some other cpu", "entries": {"x": {}}}))
    assert tune.load_cache(cache)["entries"] == {}
