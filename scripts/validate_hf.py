"""Check lowbit.model's fp32 forward pass against Hugging Face transformers.

Needs torch + transformers, which lowbit itself does not: run it from a
separate environment, e.g.

  python -m venv .cache/refenv && .cache/refenv/bin/pip install torch transformers
  PYTHONPATH=src .cache/refenv/bin/python scripts/validate_hf.py \
      --model .cache/models/Qwen2.5-0.5B --out results/e2e_validation.json

Compares (a) full-sequence logits on a 256-token window of the committed eval
text and (b) a 64-token greedy continuation, both with the HF model in fp32
(`torch_dtype=float32`, eager attention). Writes the numbers to --out.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

from lowbit import e2e
from lowbit import model as M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/e2e_validation.json")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--gen", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(4)

    tok = M.load_tokenizer(a.model)
    ids = tok.encode(e2e.eval_text()).ids[: a.tokens]

    ours = M.load(a.model, "fp32")
    lo = ours.forward(ids, logits="all")
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                                              attn_implementation="eager").eval()
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0].numpy()
    diff = np.abs(lo - ref)
    scale = np.abs(ref).max()

    def nll(lg):
        lg = lg.astype(np.float64)
        lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
        return float(np.mean(lse[:-1] - lg[np.arange(len(ids) - 1), ids[1:]]))

    prompt = ids[:64]
    g_ours = ours.generate(prompt, a.gen)
    with torch.no_grad():
        g_hf = hf.generate(torch.tensor([prompt]), max_new_tokens=a.gen, do_sample=False,
                           attention_mask=torch.ones(1, len(prompt), dtype=torch.long))
    g_hf = g_hf[0, len(prompt):].tolist()
    same = next((i for i, (x, y) in enumerate(zip(g_ours, g_hf)) if x != y), len(g_hf))
    res = {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": Path(a.model).name,
        "reference": f"transformers {transformers.__version__} / torch {torch.__version__} "
                     f"(fp32, eager attention)",
        "python": platform.python_version(),
        "tokens": len(ids),
        "logits_max_abs_diff": float(diff.max()),
        "logits_max_abs": float(scale),
        "logits_max_rel_diff": float(diff.max() / scale),
        "logits_mean_abs_diff": float(diff.mean()),
        "argmax_agreement": float(np.mean(lo.argmax(1) == ref.argmax(1))),
        "mean_nll_ours": nll(lo), "mean_nll_ref": nll(ref),
        "greedy_tokens": a.gen, "greedy_identical_prefix": same,
        "greedy_text_ours": tok.decode(g_ours),
    }
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
