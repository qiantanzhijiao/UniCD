# -*- coding: utf-8 -*-
"""
Cross-dataset training: fine-tune UniCD-PR on a SIRST dataset
(IRSTD-1K / NUAA-SIRST / NUDT-SIRST) with synthetic nonuniformity.

    python tools/train_cross.py --dataset IRSTD-1K --data-root /home/jgxy/work/cross_dataset \
        --coef-npz outputs/coef_cache_syn.npz --epochs 30 --out outputs_cross/irstd1k

Init from IRBFD joint weights (recommended) so the NUC prior is preserved;
the SIRST fine-tune adapts the detector to small-target boxes from masks.
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from unied import poly
from unied.models.unied import UniCD
from unied.datasets.irbfd import collate_irbfd
from cross_dataset.sirst import SIRSTDataset, load_coef_pool


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="IRSTD-1K")
    ap.add_argument("--data-root", default="/home/jgxy/work/cross_dataset")
    ap.add_argument("--coef-npz", default="outputs/coef_cache_syn.npz")
    ap.add_argument("--init", default=None, help="init weights (e.g. outputs_improved/joint_pr/epoch_050.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="outputs_cross/irstd1k")
    ap.add_argument("--stage", default="joint", choices=["pretrain", "joint"])
    ap.add_argument("--lambda_res", type=float, default=1.0)
    ap.add_argument("--freeze_nuc", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pool = load_coef_pool(args.coef_npz)
    ds = SIRSTDataset(args.data_root, dataset=args.dataset, split="train",
                      coef_pool=pool, size=args.size, hflip=0.5, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2,
                        collate_fn=collate_irbfd, drop_last=True)
    print(f"[cross] {args.dataset} train={len(ds)} batch={args.batch} device={device}")

    residual = {"enabled": True, "gate": True}
    model = UniCD(degree=poly.DEFAULT_DEGREE, lambda_tebs=0.01,
                  residual=residual, lambda_res=args.lambda_res)
    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        state = ck["model"] if "model" in ck else ck
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[cross] init from {args.init} (missing={len(missing)} unexpected={len(unexpected)})")
    model.to(device)

    if args.freeze_nuc:
        for p in model.nuc.parameters():
            p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = [p for p in model.parameters()]

    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    log_f = open(out / "train_log.csv", "w")
    log_f.write("epoch,step,loss,det,cls,reg,tebs,br,res,sec\n")

    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for batch in loader:
            y = batch["y"].to(device)
            c = batch["c"].to(device)
            boxes = batch["boxes"].to(device)
            img_size = (y.shape[2], y.shape[3])
            if args.stage == "pretrain":
                coef_gt = batch["coef_gt"].to(device)
                losses = model.forward_nuc_pretrain(y, coef_gt, c)
            else:
                losses = model.forward_joint(y, c, boxes, img_size,
                                             use_br=True, lambda_tebs=0.01,
                                             lambda_res=args.lambda_res)
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                dt = time.time() - t0
                print(f"[cross] ep={ep} step={step} loss={losses['total'].item():.4f} "
                      f"det={losses.get('det', 0):.4f} res={losses.get('res', 0):.4f} "
                      f"sec/iter={dt/20:.3f}")
                log_f.write(f"{ep},{step},{losses['total'].item():.6f},"
                            f"{losses.get('det', torch.zeros(1)).item():.6f},"
                            f"{losses.get('cls', torch.zeros(1)).item():.6f},"
                            f"{losses.get('reg', torch.zeros(1)).item():.6f},"
                            f"{losses.get('tebs', torch.zeros(1)).item():.6f},"
                            f"{losses.get('br', torch.zeros(1)).item():.6f},"
                            f"{losses.get('res', torch.zeros(1)).item():.6f},"
                            f"{dt/20:.3f}\n")
                log_f.flush()
                t0 = time.time()
        torch.save({"model": model.state_dict(), "epoch": ep}, out / f"epoch_{ep:03d}.pt")

    torch.save({"model": model.state_dict(), "epoch": args.epochs}, out / "final.pt")
    log_f.close()
    print(f"[cross] done -> {out}/final.pt")


if __name__ == "__main__":
    main()
