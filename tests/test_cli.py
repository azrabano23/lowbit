import json
import subprocess
import sys


def run(*args):
    r = subprocess.run([sys.executable, "-m", "lowbit.cli", *args], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_info():
    d = json.loads(run("info"))
    assert d["active_isa"] in ("scalar", "avx2", "avx512")
    assert set(d["effective"]) == {"w4a16", "mxfp4", "w4a8"}


def test_tune_cli(tmp_path):
    out = run("tune", "--shape", "2x32x256", "--kernel", "w4a8", "--threads", "1", "--reps", "1",
              "--cache", str(tmp_path / "t.json"))
    assert "w4a8 2x32x256" in out and (tmp_path / "t.json").exists()


def test_bench_and_report_tiny(tmp_path):
    out = run("bench", "--shape", "1x64x256", "--threads", "1", "--min-time", "0.01",
              "--no-cold", "--out", str(tmp_path))
    res = json.loads((tmp_path / "bench.json").read_text())
    methods = {r["method"] for r in res["rows"]}
    assert "numpy fp32" in methods and "w4a16 scalar" in methods
    assert all(r["max_rel_err"] < 0.05 for r in res["rows"])
    assert (tmp_path / "RESULTS.md").read_text().startswith("# lowbit benchmark results")
    assert "wrote" in out
