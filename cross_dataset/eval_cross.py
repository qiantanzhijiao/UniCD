# -*- coding: utf-8 -*-
"""
Cross-dataset evaluation: NUC quality (PSNR/SSIM) + detection (Pd/Fa/nIoU).

    python tools/eval_cross.py --dataset IRSTDD-1K ... actually IRSTD-1K \
        --data-root /home/jgxy/work/cross_dataset \
        --ckpt outputs_cross/irstd1k/final.pt --coef-npz outputs/coef_cache_syn.npz

Evaluated on the SIRST test split with synthetic nonuniformity injected
(fixed seed so all methods see identical degradations).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from unied import poly
from unied.models.unied import UniCD
from unied.utils.metrics import psnr, ssim
from cross_dataset.sirst import SIRSTDataset, load_coef_pool
from cross_dataset.sirst_metrics import SIRSTMetrics
from tools.classical_nuc_baseline import moment_matching_correction, two_point_correction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="IRSTD-1K")
    ap.add_argument("--data-root", default="/home/jgxy/work/cross_dataset")
    ap.add_argument("--coef-npz", default="outputs/coef_cache_syn.npz")
    ap.add_argument("--ckpt", default=None, help="UniCD-PR weights (required for --method unicd)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--method", default="unicd", choices=["unicd", "none", "mm", "tpc"],
                    help="unicd=UniCD-PR; none=no correction; mm=moment matching; tpc=two-point")
    ap.add_argument("--baseline", action="store_true",
                    help="(legacy flag, same as --method none)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = load_coef_pool(args.coef_npz)
    # fixed seed -> identical synthetic degradations for every method
    ds = SIRSTDataset(args.data_root, dataset=args.dataset, split="test",
                      coef_pool=pool, size=args.size, hflip=0.0,
                      stripe_p=0.0, seed=123)

    model = None
    if args.method == "unicd":
        model = UniCD(degree=poly.DEFAULT_DEGREE, residual={"enabled": True, "gate": True})
        ck = torch.load(args.ckpt, map_location="cpu")
        state = ck["model"] if "model" in ck else ck
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

    met = SIRSTMetrics(score_thr=args.score_thr)
    psnrs, ssims = [], []

    for i in range(len(ds)):
        item = ds[i]
        y = item["y"].unsqueeze(0).to(device)      # [1,3,H,W]
        c = item["c"].numpy()                      # [3,H,W]
        boxes_gt = item["boxes"].numpy()
        h = w = args.size

        with torch.no_grad():
            if args.method == "unicd":
                r = model.correct(y)
            else:
                r = y
        r_np = r.squeeze(0).cpu().numpy()          # [3,H,W]

        if args.method == "mm":
            r_np = np.stack([moment_matching_correction(r_np[k]) for k in range(3)], axis=0)
        elif args.method == "tpc":
            r_np = np.stack([two_point_correction(r_np[k], c[k]) for k in range(3)], axis=0)

        psnrs.append(psnr(c, r_np))
        ssims.append(ssim(c, r_np))

        # rasterize GT boxes to mask (detection-from-boxes protocol)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for bx in boxes_gt:
            x1, y1, x2, y2 = [int(round(v)) for v in bx]
            gt_mask[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = 1

        if args.method == "unicd":
            with torch.no_grad():
                boxes, scores = model.detect(y, score_thr=args.score_thr)[0]
            met.update(gt_mask, boxes.numpy(), scores.numpy())
        else:
            # classical corrections share the 3.5-sigma threshold detector
            gray = r_np.mean(axis=0)
            thr = gray.mean() + 3.5 * gray.std()
            ys, xs = np.nonzero(gray > thr)
            det_boxes = []
            for yy, xx in zip(ys, xs):
                det_boxes.append([max(0, xx - 2), max(0, yy - 2), min(w - 1, xx + 2), min(h - 1, yy + 2)])
            met.update(gt_mask, np.asarray(det_boxes, dtype=np.float32) if det_boxes else np.zeros((0, 4), np.float32),
                       np.ones(len(det_boxes)))

    det = met.compute()
    result = {
        "dataset": args.dataset,
        "method": args.method,
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "Pd": det["Pd"],
        "Fa": det["Fa"],
        "nIoU": det["nIoU"],
        "n_images": det["n_images"],
        "n_gt": det["n_gt"],
    }
    print(json.dumps(result, indent=2))
    out = args.out or f"outputs_cross/eval_{args.dataset}_{args.method}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
