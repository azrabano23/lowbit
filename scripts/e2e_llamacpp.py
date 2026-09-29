"""End-to-end tokens/s of llama.cpp (`llama-bench`) on the same model, next to ours.

Runs llama-bench on one or more GGUF files (-p PP -n TG, each thread count),
waits for an idle machine before each run, and writes results/e2e_llamacpp.json
with the llama.cpp rows plus the matching lowbit rows copied from
results/e2e.json (pp64 = prompt prefill, pp512, tg128 = decode after the prompt).

How the GGUFs were made (llama.cpp built as in scripts/compare_llamacpp.py, plus
`ninja -C build llama-bench llama-quantize`):
  python convert_hf_to_gguf.py <Qwen2.5-0.5B dir> --outtype f32 --outfile qwen-f32.gguf
  build/bin/llama-quantize qwen-f32.gguf qwen-q4_0.gguf Q4_0
  build/bin/llama-quantize --pure qwen-f32.gguf qwen-q4_0-pure.gguf Q4_0

Usage: python scripts/e2e_llamacpp.py --bench build/bin/llama-bench \
          --gguf f32=qwen-f32.gguf --gguf q4_0=qwen-q4_0.gguf [--out results]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from lowbit import bench as B


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--gguf", action="append", required=True, help="label=path")
    ap.add_argument("--threads", type=int, action="append")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    threads = a.threads or [1, 4]
    rows, commit, flags = [], None, None
    for spec in a.gguf:
        label, path = spec.split("=", 1)
        for th in threads:
            B.wait_idle(0.9, 900)
            cmd = [a.bench, "-m", path, "-t", str(th), "-p", "64,512", "-n", "128",
                   "-r", str(a.reps), "-o", "json"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                raise SystemExit(r.stderr)
            flags = " ".join(cmd[3:]).replace(path, "<gguf>")
            row = {"impl": "llama.cpp", "type": label, "threads": th}
            for t in json.loads(r.stdout):
                commit = t.get("build_commit")
                key = f"pp{t['n_prompt']}" if t["n_gen"] == 0 else f"tg{t['n_gen']}"
                row[key] = t["avg_ts"]
                row[key + "_stddev"] = t["stddev_ts"]
                row["model_type"], row["model_size"] = t.get("model_type"), t.get("model_size")
            rows.append(row)
            print(row)
    ours = Path(a.out) / "e2e.json"
    if ours.exists():
        for r in json.loads(ours.read_text())["speed"]:
            if r["threads"] in threads:
                rows.append({"impl": "lowbit (numpy model)", "type": r["backend"],
                             "threads": r["threads"], "pp64": r["prefill_tok_s"],
                             "pp512": r.get("pp_tok_s"), "tg128": r["decode_tok_s"]})
    meta = {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "llama_cpp_commit": commit, "tests": ["pp64", "pp512", "tg128"],
        "gguf": "Qwen2.5-0.5B converted with convert_hf_to_gguf.py --outtype f32, "
                "then llama-quantize",
        "note": (f"llama.cpp build `{commit}` (CMake Release, -DGGML_NATIVE=ON, OpenMP), CPU "
                 f"backend with its default extra buffer types: `llama-bench -v` shows the "
                 f"Q4_0 matrices (and the Q8_0 embedding) repacked into ggml's AMX buffer, "
                 f"i.e. llama.cpp runs its AMX-INT8 kernels here. "
                 f"`llama-bench {flags}`, mean of {a.reps} reps, idle-wait before each run. "
                 "llama.cpp's tg128 starts from an empty context; ours decodes 128 tokens after "
                 "the 64-token prompt. `q4_0` is llama-quantize's default Q4_0 mix (the tied "
                 "token embedding/output matrix at Q8_0), `q4_0-pure` quantizes every matrix "
                 "to Q4_0 (block 32, fp16 scale) — closer to our all-4-bit setup."),
    }
    p = Path(a.out) / "e2e_llamacpp.json"
    p.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
