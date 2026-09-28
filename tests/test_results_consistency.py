"""The README numbers must be exactly what results/bench.json renders to."""
import json
from pathlib import Path

import pytest

from lowbit import report

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "bench.json"


@pytest.mark.skipif(not BENCH.exists(), reason="no committed benchmark results")
def test_readme_block_matches_results():
    text = (ROOT / "README.md").read_text()
    block = text.split(report.BEGIN, 1)[1].split(report.END, 1)[0].strip()
    assert block == report.readme_block(ROOT / "results").strip()


@pytest.mark.skipif(not BENCH.exists(), reason="no committed benchmark results")
def test_results_md_matches_results():
    res = json.loads(BENCH.read_text())
    tc = ROOT / "results" / "tune_cache.json"
    cache = json.loads(tc.read_text()) if tc.exists() else None
    lc = ROOT / "results" / "llamacpp.json"
    llama = json.loads(lc.read_text()) if lc.exists() else None
    assert (ROOT / "results" / "RESULTS.md").read_text() == report.full(res, cache, llama)


@pytest.mark.skipif(not BENCH.exists(), reason="no committed benchmark results")
def test_benchmarked_outputs_were_correct():
    rows = json.loads(BENCH.read_text())["rows"]
    for r in rows:
        lim = 0.05 if r["format"] == "w4a8" else 1e-4  # w4a8 includes int8 activation error
        assert r["max_rel_err"] is not None and r["max_rel_err"] < lim, r["method"]
