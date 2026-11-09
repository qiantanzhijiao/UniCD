# -*- coding: utf-8 -*-
"""
Evaluate UAV detection: P, R (at a confidence threshold) and AP50.

    python tools/eval_det.py --config configs_improved/eval_pr.yaml \
        --ckpt outputs_improved/joint_pr/epoch_050.pt --mode syn --split test

`--mode` selects the dataset: syn (IRBFD-syn test) or real (IRBFD-real test).
Use --data-root to override the dataset roots (same semantics as train.py).
The config must match the residual setting used at training time.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from unied import load_config
from unied.datasets.irbfd import IRBFDDataset, collate_irbfd
from unied.models.unied import UniCD
from unied.utils.metrics import compute_pr, compute_ap50


def resolve_data_paths(cfg, data_root):
    if not data_root:
        return
    root = str(data_root).rstrip("/\\")
    cfg.data.syn_root = root + "/IRBFD-syn/VOC2007"
    cfg.data.real_root = root + "/IRBFD-real/VOC2007"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="syn", choices=["syn", "real"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="0 = evaluate all images")
    ap.add_argument("--data-root", default=None, help="override dataset roots")
    args = ap.parse_args()
    cfg = load_config(args.config)
    resolve_data_paths(cfg, args.data_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UniCD(degree=cfg.model.get("degree", 3), residual=cfg.model.get("residual"))
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    root = cfg.data.syn_root if args.mode == "syn" else cfg.data.real_root
    ds = IRBFDDataset(
        voc_root=root,
        split=args.split,
        mode=args.mode,
        coef_cache=None,  # detection evaluation needs no coefficients
        scale=cfg.data.get("scale", 1.0),
        hflip=0.0,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_irbfd)

    preds, gts = [], []
    n_img = 0
    with torch.no_grad():
        for batch in loader:
            if args.limit and n_img >= args.limit:
                break
            n_img += 1
            y = batch["y"].to(device)
            boxes, scores = model.detect(y, score_thr=args.score_thr)[0]
            preds.append((boxes.numpy(), scores.numpy()))
            gts.append(batch["boxes"][0].numpy())

    prec, rec, tp, fp, fn, n_pred, n_gt = compute_pr(preds, gts, iou_thr=0.5, score_thr=args.score_thr)
    ap50 = compute_ap50(preds, gts, iou_thr=0.5)

    res = {
        "mode": args.mode,
        "split": args.split,
        "score_thr": args.score_thr,
        "n_images": len(preds),
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "P": round(prec, 4),
        "R": round(rec, 4),
        "AP50": round(ap50, 4),
    }
    out_dir = Path(cfg.output.save_dir) / "eval_det"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"metrics_{args.mode}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
