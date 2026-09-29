"""lowbit.e2e: pinned eval text, the `lowbit e2e --tiny` CLI, committed results."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lowbit import e2e
from lowbit import model as M

ROOT = Path(__file__).resolve().parents[1]


def test_eval_text_pinned():
    t = e2e.eval_text()
    assert t.startswith(" = Robert Boulter = ") and len(t) == 88203


def test_e2e_cli_tiny(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n{e2e.BEGIN}\nold\n{e2e.END}\ntail\n")
    r = subprocess.run([sys.executable, "-m", "lowbit.cli", "e2e", "--tiny", "--threads", "1",
                        "--reps", "1", "--prompt", "8", "--decode", "4", "--pp", "16",
                        "--eval-tokens", "200", "--ctx", "64", "--out", str(tmp_path),
                        "--readme", str(readme)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "e2e.json").read_text())
    assert [q["backend"] for q in res["quality"]] == list(M.BACKENDS)
    assert {s["backend"] for s in res["speed"]} == set(M.BACKENDS)
    for s in res["speed"]:
        assert s["decode_tok_s"] > 0 and abs(sum(s["breakdown_decode"].values()) - 1) < 0.01
    for q in res["quality"][1:]:
        assert q["top1_agree_fp32"] >= 0.8
    text = readme.read_text()
    assert "old" not in text and text.endswith(f"{e2e.END}\ntail\n") and "| mxfp4 |" in text


E2E = ROOT / "results" / "e2e.json"


@pytest.mark.skipif(not E2E.exists(), reason="no committed e2e results")
def test_readme_e2e_block_matches_results():
    text = (ROOT / "README.md").read_text()
    block = text.split(e2e.BEGIN, 1)[1].split(e2e.END, 1)[0].strip()
    assert block == e2e.readme_block(ROOT / "results").strip()


@pytest.mark.skipif(not E2E.exists(), reason="no committed e2e results")
def test_committed_e2e_results_sane():
    res = json.loads(E2E.read_text())
    q = {r["backend"]: r for r in res["quality"]}
    assert set(M.BACKENDS) <= set(q)
    assert q["fp32"]["ppl"] < 30  # an English-text sanity bound for a working LM
    for be in set(q) - {"fp32"}:
        assert q["fp32"]["ppl"] < q[be]["ppl"] < 2 * q["fp32"]["ppl"]
        assert q[be]["top1_agree_fp32"] > 0.5
    for r in res["speed"]:
        assert r["decode_tok_s"] > 0 and r["prefill_tok_s"] > 0
    v = ROOT / "results" / "e2e_validation.json"
    if v.exists():
        val = json.loads(v.read_text())
        assert val["logits_max_rel_diff"] < 1e-4 and val["argmax_agreement"] == 1.0
