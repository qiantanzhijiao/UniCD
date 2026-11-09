# -*- coding: utf-8 -*-
"""
Evaluate the NUC module: PSNR / SSIM on IRBFD-syn test (R vs C), plus SCRG.

    python tools/eval_nuc.py --config configs_improved/eval_pr.yaml \
        --ckpt outputs_improved/joint_pr/epoch_050.pt

Saves corrected + bias-field sample images under the output dir.
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
from PIL import Image

from unied import load_config
from unied.datasets.irbfd import IRBFDDataset, collate_irbfd
from unied.models.unied import UniCD
from unied.utils.metrics import psnr, ssim, scrg


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
    ap.add_argument("--split", default="test")
    ap.add_argument("--save-samples", type=int, default=4)
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

    ds = IRBFDDataset(
        voc_root=cfg.data.syn_root,
        split=args.split,
        mode="syn",
        coef_cache=None,  # NUC evaluation compares R vs C; no coefficients needed
        scale=cfg.data.get("scale", 1.0),
        hflip=0.0,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_irbfd)

    psnr_acc, ssim_acc, n = 0.0, 0.0, 0
    scrg_acc = 0.0
    samples = []
    with torch.no_grad():
        for batch in loader:
            if args.limit and n >= args.limit:
                break
            y = batch["y"].to(device)
            c = batch["c"].numpy()[0]
            r = model.correct(y).clamp(0, 1).cpu().numpy()[0]
            r_np = r.transpose(1, 2, 0)
            c_np = c.transpose(1, 2, 0)
            psnr_acc += psnr(c_np, r_np)
            ssim_acc += ssim(c_np, r_np)
            boxes = batch["boxes"][0].numpy()
            y_np = y.cpu().numpy()[0].transpose(1, 2, 0)
            scrg_acc += scrg(y_np, r_np, boxes)
            n += 1
            if len(samples) < args.save_samples:
                samples.append((batch["stems"][0], y_np, c_np, r_np))

    out_dir = Path(cfg.output.save_dir) / "eval_nuc"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (stem, y_np, c_np, r_np) in enumerate(samples):
        for tag, arr in [("degraded", y_np), ("clear", c_np), ("corrected", r_np)]:
            Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(
                out_dir / f"{i:02d}_{stem}_{tag}.png"
            )

    res = {
        "split": args.split,
        "n_images": n,
        "PSNR": round(psnr_acc / max(n, 1), 4),
        "SSIM": round(ssim_acc / max(n, 1), 4),
        "SCRG": round(scrg_acc / max(n, 1), 4),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"samples saved under {out_dir}")


if __name__ == "__main__":
    main()
