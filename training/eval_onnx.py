"""Evaluate the exported ONNX model through the same code path the agent uses (vision.classify).

Also measures selective prediction: how accurate is the model on the photos where it
reports "confident"? That's what matters for a farmer-facing tool.

    python training/eval_onnx.py --test ../datasets/PlantDoc-Dataset-master/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kisan_agent import vision

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/onnx_eval.json"))
    args = ap.parse_args()

    rows = []
    for d in sorted(p for p in args.test.iterdir() if p.is_dir()):
        for img in sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS):
            out = vision.classify(img.read_bytes(), top_k=3)
            if "error" in out:
                print(f"  unreadable: {img.name}")
                continue
            guesses = out["top_guesses"]
            rows.append({"true": d.name, "top3": [g["label"] for g in guesses], "conf": guesses[0]["confidence"]})

    n = len(rows)
    report = {
        "n": n,
        "top1": round(sum(r["true"] == r["top3"][0] for r in rows) / n, 4),
        "top3": round(sum(r["true"] in r["top3"] for r in rows) / n, 4),
        "selective": [],
    }
    for th in (0.4, 0.5, vision.CONFIDENT, 0.7):
        kept = [r for r in rows if r["conf"] >= th]
        report["selective"].append({
            "threshold": th,
            "coverage": round(len(kept) / n, 4),
            "accuracy_when_confident": round(sum(r["true"] == r["top3"][0] for r in kept) / max(len(kept), 1), 4),
        })

    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
