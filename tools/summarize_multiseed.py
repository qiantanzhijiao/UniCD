# -*- coding: utf-8 -*-
"""Aggregate per-seed reduced-budget evaluation JSONs into mean +/- std.

The eval scripts print a JSON object followed by free text (eval_nuc prints a
"samples saved under ..." line), so each file is parsed with raw_decode from
its first '{'.  Any seed whose three files are not all present is skipped,
which lets this script be called after every seed and still produce a valid
partial summary.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_jsonish(path):
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")
    i = txt.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(txt[i:])
        return obj
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.dir)
    tags = sorted({re.sub(r"_(det_syn|det_real|nuc)\.json$", "", p.name)
                   for p in d.glob("seed*_*.json")})
    per_seed = {}
    for tag in tags:
        syn = load_jsonish(d / f"{tag}_det_syn.json")
        real = load_jsonish(d / f"{tag}_det_real.json")
        nuc = load_jsonish(d / f"{tag}_nuc.json")
        if not (syn and real and nuc):
            continue
        per_seed[tag] = {
            "AP50_syn": syn.get("AP50"),
            "AP50_real": real.get("AP50"),
            "PSNR": nuc.get("PSNR"),
            "SSIM": nuc.get("SSIM"),
            "P_syn": syn.get("P"),
            "R_syn": syn.get("R"),
            "n_images_syn": syn.get("n_images"),
            "n_images_real": real.get("n_images"),
        }

    keys = ("AP50_syn", "AP50_real", "PSNR", "SSIM")
    summary = {
        "protocol": "matched 10-epoch joint training, 3000 iters/epoch, "
                    "common pretrain init, score_thr=0.5, full test splits",
        "n_seeds": len(per_seed),
        "seeds": sorted(per_seed),
        "per_seed": per_seed,
        "mean_std": {},
    }
    for k in keys:
        vals = [per_seed[t][k] for t in sorted(per_seed)
                if per_seed[t].get(k) is not None]
        if vals:
            summary["mean_std"][k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=0)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "values": [round(float(v), 4) for v in vals],
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["mean_std"], indent=2))


if __name__ == "__main__":
    main()
