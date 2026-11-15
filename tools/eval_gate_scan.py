# -*- coding: utf-8 -*-
"""
Gate causality scan on a SINGLE checkpoint.

For the learned adaptive gate and for gate forced to constants
g in {0, 0.3, 0.5, 0.7, 1.0}, evaluate detection AP50 and (on syn, where a
clean reference exists) PSNR/SSIM. All rows share identical weights, so any
difference is caused solely by the gate value -- this removes the confound in
the earlier table where g=0/g=1 came from different checkpoints.

Usage:
  python tools/eval_gate_scan.py --config configs_improved/eval_pr.yaml \
      --ckpt outputs_improved/joint_pr/epoch_050.pt --mode syn \
      --data-root /home/jgxy/work/UniCD --out logs/gate_scan_syn.json
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
from unied.utils.metrics import compute_pr, compute_ap50, psnr, ssim


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
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, use an evenly spaced deterministic subset "
                         "of this many test images (all gate settings share "
                         "the identical subset).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    resolve_data_paths(cfg, args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UniCD(degree=cfg.model.get("degree", 3),
                  residual=cfg.model.get("residual"))
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    root = cfg.data.syn_root if args.mode == "syn" else cfg.data.real_root
    ds = IRBFDDataset(voc_root=root, split="test", mode=args.mode,
                      coef_cache=None, scale=cfg.data.get("scale", 1.0),
                      hflip=0.0)
    if args.limit > 0 and args.limit < len(ds):
        idx = np.linspace(0, len(ds) - 1, args.limit).round().astype(int)
        idx = sorted(set(int(i) for i in idx))
        ds = torch.utils.data.Subset(ds, idx)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_irbfd)

    orig_gate = model.nuc.gate  # bound method for the learned gate
    has_clean = (args.mode == "syn")
    print("dataset size: %d images (limit=%d)" % (len(loader), args.limit),
          flush=True)

    results = {}
    settings = [("g=0", 0.0), ("g=0.3", 0.3), ("g=0.5", 0.5),
                ("g=0.7", 0.7), ("g=1", 1.0), ("learned", None)]

    for name, gval in settings:
        if gval is None:
            model.nuc.gate = orig_gate
        else:
            def fixed_gate(f_fused, v=gval):
                return torch.full((f_fused.shape[0], 1, 1, 1), v,
                                  device=f_fused.device, dtype=f_fused.dtype)
            model.nuc.gate = fixed_gate

        preds, gts = [], []
        psnr_acc = ssim_acc = 0.0
        n_img = 0
        with torch.no_grad():
            # stream batches: re-iterate the loader per gate value to keep RAM
            # bounded (pre-loading 2880 full-size tensors can exhaust 8 GB hosts)
            for batch in loader:
                y = batch["y"].to(device)
                boxes, scores = model.detect(y, score_thr=args.score_thr)[0]
                preds.append((boxes.numpy(), scores.numpy()))
                gts.append(batch["boxes"][0].numpy())
                if has_clean:
                    c = batch["c"].numpy()[0].transpose(1, 2, 0)
                    r = model.correct(y).clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
                    psnr_acc += psnr(c, r)
                    ssim_acc += ssim(c, r)
                n_img += 1

        prec, rec, tp, fp, fn, n_pred, n_gt = compute_pr(
            preds, gts, iou_thr=0.5, score_thr=args.score_thr)
        ap50 = compute_ap50(preds, gts, iou_thr=0.5)
        rec_json = {
            "gate": name,
            "mode": args.mode,
            "score_thr": args.score_thr,
            "n_images": n_img,
            "P": round(float(prec), 4),
            "R": round(float(rec), 4),
            "AP50": round(float(ap50), 4),
            "n_pred": int(n_pred),
        }
        if has_clean:
            rec_json["PSNR"] = round(float(psnr_acc / n_img), 4)
            rec_json["SSIM"] = round(float(ssim_acc / n_img), 4)
        results[name] = rec_json
        print(json.dumps(rec_json), flush=True)

        # incremental persistence so partial results survive interruption
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("GATE_SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
