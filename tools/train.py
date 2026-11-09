# -*- coding: utf-8 -*-
"""
UniCD training entry point (all three stages), including the improved
parametric + residual NUC (scheme B) and perturbation augmentation (scheme A).

    python tools/train.py --config configs/pretrain_nuc.yaml     # stage 1
    python tools/train.py --config configs/joint_train.yaml      # stage 2
    python tools/train.py --config configs/finetune_real.yaml    # stage 3
    python tools/train.py --config configs/smoke_cpu.yaml        # quick CPU check

Improved scheme (see configs_improved/):
    python tools/train.py --config configs_improved/pretrain_pr.yaml
    python tools/train.py --config configs_improved/joint_pr.yaml
    python tools/train.py --config configs_improved/finetune_real_pr.yaml

Stages:
  * pretrain      -- NUC only, L_cor = MAE(a_hat, a) on IRBFD-syn (Eq. 5);
                     + L_res on the residual branch when enabled
  * joint         -- L_uni = L_det + L_BR [+ lambda_res*L_res] on IRBFD-syn
                     (Eq. 11); lambda_TEBS is 1.0 for the first
                     `lambda_switch_epoch` epochs, then 0.01
  * finetune_real -- L_det on IRBFD-real, NUC frozen (UniCD* setup in the paper)

Optimizer: Adam, lr 1e-3, weight decay 1e-4.
Augmentation: random h-flip; optional perturbation augmentation (scheme A)
applied to the degraded input in the joint / finetune_real stages.

`--data-root` overrides the dataset roots in the config (useful when the same
config file is used on a machine where the data lives elsewhere, e.g.
    python tools/train.py --config configs/joint_train.yaml --data-root E:/work/UniCD
resolves IRBFD-syn/VOC2007, IRBFD-real/VOC2007 and the coef cache under it).
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from unied import load_config
from unied.augment import PerturbationAug
from unied.datasets.irbfd import IRBFDDataset, collate_irbfd
from unied.models.unied import UniCD
from unied.utils.logger import TrainLogger

STAGES = ("pretrain", "joint", "finetune_real")


def _val(d, key):
    """Safely read a (possibly missing) tensor value from a loss dict."""
    v = d.get(key)
    return float(v) if isinstance(v, torch.Tensor) else 0.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_data_paths(cfg, data_root):
    """Override the dataset roots with a single root directory if requested."""
    if not data_root:
        return
    root = str(data_root).rstrip("/\\")
    cfg.data.syn_root = root + "/IRBFD-syn/VOC2007"
    cfg.data.real_root = root + "/IRBFD-real/VOC2007"
    if cfg.data.get("coef_cache"):
        cfg.data.coef_cache = root + "/UniCD-repro/outputs/coef_cache_syn.npz"


def build_loader(cfg, split: str, mode: str, scale: float, hflip: float):
    root = cfg.data.syn_root if mode == "syn" else cfg.data.real_root
    ds = IRBFDDataset(
        voc_root=root,
        split=split,
        mode=mode,
        coef_cache=cfg.data.get("coef_cache") if mode == "syn" else None,
        scale=scale,
        hflip=hflip,
    )
    nw = cfg.data.get("num_workers", 0)
    return DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=(split == "train"),
        num_workers=nw,
        collate_fn=collate_irbfd,
        drop_last=(split == "train"),
        pin_memory=(nw > 0),
        persistent_workers=(nw > 0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None,
                    help="override IRBFD roots; e.g. E:/work/UniCD or /root/work/UniCD")
    args = ap.parse_args()
    cfg = load_config(args.config)
    resolve_data_paths(cfg, args.data_root)

    stage = cfg.train.stage
    assert stage in STAGES, f"stage must be one of {STAGES}"
    set_seed(cfg.train.get("seed", 42))
    torch.backends.cudnn.benchmark = True  # speed up fixed-size convolutions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UniCD] stage={stage} device={device} config={cfg.config_path}")

    residual_cfg = cfg.model.get("residual")
    lambda_res = cfg.train.get("lambda_res", 1.0)
    model = UniCD(
        degree=cfg.model.get("degree", 3),
        lambda_tebs=cfg.train.get("lambda_tebs_after", 0.01),
        residual=residual_cfg,
        lambda_res=lambda_res,
    )
    model.to(device)
    if residual_cfg and residual_cfg.get("enabled", False):
        print(f"[UniCD] residual NUC enabled (gate={bool(residual_cfg.get('gate', False))})")

    perturb_cfg = cfg.train.get("perturb")
    perturb = PerturbationAug(**perturb_cfg) if isinstance(perturb_cfg, dict) else PerturbationAug(enabled=False)
    if stage in ("joint", "finetune_real"):
        print(f"[UniCD] perturbation augmentation enabled={perturb.enabled} p={perturb.p} types={perturb.types}")

    # pick the loader for this stage
    if stage == "pretrain":
        loader = build_loader(cfg, "train", "syn", cfg.data.get("scale", 1.0), cfg.data.get("hflip", 0.5))
        param_groups = [p for n, p in model.named_parameters() if n.startswith("nuc.")]
    elif stage == "joint":
        loader = build_loader(cfg, "train", "syn", cfg.data.get("scale", 1.0), cfg.data.get("hflip", 0.5))
        param_groups = [p for p in model.parameters()]
    else:  # finetune_real
        loader = build_loader(cfg, "train", "real", cfg.data.get("scale", 1.0), cfg.data.get("hflip", 0.5))
        if cfg.train.get("freeze_nuc", True):
            for p in model.nuc.parameters():
                p.requires_grad = False
        param_groups = [p for p in model.parameters() if p.requires_grad]

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[UniCD] total params={n_params / 1e6:.4f}M, trainable={n_train / 1e6:.4f}M")

    save_dir = Path(cfg.output.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(str(save_dir))

    optimizer = torch.optim.Adam(param_groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    start_epoch = 0
    if cfg.train.get("resume"):
        ckpt = torch.load(cfg.train.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = 0  # resume loads weights only; each stage counts its own epochs from 0
        print(f"[UniCD] resumed weights from {cfg.train.resume}")

    epochs = cfg.train.epochs
    lambda_switch = cfg.train.get("lambda_switch_epoch", 20)
    lam_first = cfg.train.get("lambda_tebs_first", 1.0)
    lam_after = cfg.train.get("lambda_tebs_after", 0.01)
    use_br = cfg.train.get("use_br", stage == "joint")
    log_every = cfg.output.get("log_every", 10)
    save_every = cfg.output.get("save_every", 5)
    max_iters = cfg.output.get("max_iters_per_epoch", 0)  # 0 = full epoch

    img_size = None
    step_global = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        lam = lam_first if epoch < lambda_switch else lam_after
        model.lambda_tebs = lam
        print(f"[UniCD] epoch {epoch + 1}/{epochs}  lambda_TEBS={lam}")

        epoch_loss = 0.0
        n_steps = 0
        t_start = time.time()
        for step, batch in enumerate(loader):
            if max_iters and step >= max_iters:
                break
            n_steps = step + 1
            y = batch["y"].to(device)
            boxes = batch["boxes"].to(device)
            if img_size is None:
                img_size = (y.shape[2], y.shape[3])
                model.set_image_size(*img_size, device=device)

            # scheme A: perturbation augmentation on the degraded input
            # (not in pre-training: the coefficient targets are fitted on the
            #  unperturbed bias field)
            if stage in ("joint", "finetune_real"):
                y = perturb(y)

            if stage == "pretrain":
                losses = model.forward_nuc_pretrain(
                    y, batch["coef_gt"].to(device),
                    c=batch["c"].to(device) if batch.get("c") is not None else None,
                    lambda_res=lambda_res,
                )
            elif stage == "joint":
                c = batch["c"].to(device)
                losses = model.forward_joint(
                    y, c, boxes, img_size, use_br=use_br, lambda_tebs=lam,
                    lambda_res=lambda_res, gate_mode="learn",
                )
            else:
                losses = model.forward_det_only(y, boxes, img_size)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(param_groups, 10.0)
            optimizer.step()
            step_global += 1
            epoch_loss += float(losses["total"].item())

            if step % log_every == 0 or step == len(loader) - 1:
                sec_iter = (time.time() - t_start) / max(step + 1, 1)
                logger.log(
                    epoch=epoch + 1, step=step_global,
                    loss=losses["total"].item(),
                    det=_val(losses, "det"),
                    cls=_val(losses, "cls"),
                    reg=_val(losses, "reg"),
                    tebs=_val(losses, "tebs"),
                    br=_val(losses, "br"),
                    cor=_val(losses, "cor"),
                    res=_val(losses, "res"),
                    **{"sec/iter": round(sec_iter, 3)},
                )

        avg = epoch_loss / max(n_steps, 1)
        print(f"[UniCD] epoch {epoch + 1} avg loss {avg:.4f}")

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_path = save_dir / f"epoch_{epoch + 1:03d}.pt"
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "config": cfg.config_path},
                ckpt_path,
            )
            print(f"[UniCD] checkpoint saved -> {ckpt_path}")

    print("[UniCD] training finished")


if __name__ == "__main__":
    main()
