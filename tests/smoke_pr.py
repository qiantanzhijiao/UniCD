# -*- coding: utf-8 -*-
"""Smoke test for the improved scheme (B: residual NUC + gate, A: perturbation).

Random data only; checks the residual branch, the detection-driven gate, the
L_res supervision and the perturbation augmentation end to end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from unied.augment import PerturbationAug
from unied.models.unied import UniCD

torch.manual_seed(0)

RESIDUAL = {"enabled": True, "gate": True}

print("== build model (residual + gate) ==")
model = UniCD(degree=3, residual=RESIDUAL, lambda_res=1.0)
n_params = sum(p.numel() for p in model.parameters())
print(f"total params: {n_params / 1e6:.4f} M  (baseline was ~0.46M)")
assert model.use_residual

H, W = 512, 640
y = torch.rand(2, 3, H, W)
c = torch.rand(2, 3, H, W)
boxes = torch.tensor([[[100.0, 50, 130, 90], [0, 0, 0, 0]],
                      [[200.0, 300, 220, 310], [400, 100, 430, 150]]])
model.set_image_size(H, W)

print("== bias decomposition ==")
b = model.bias_field(y)
print("B_total:", b.shape)
assert b.shape == (2, 1, H, W)

print("== correct (gate active) ==")
r = model.correct(y)
print("corrected shape:", r.shape)

print("== perturbation augmentation ==")
aug = PerturbationAug(enabled=True, p=1.0, types=("hres", "stripe", "gain"), strength=1.0)
y_p = aug(y)
print("perturbed range:", float(y_p.min()), float(y_p.max()))
assert y_p.shape == y.shape
assert torch.isfinite(y_p).all()
d = (y_p - y).abs().mean().item()
print("mean |delta| under perturbation:", round(d, 4))
assert d > 0.0  # perturbation actually changed the input

print("== forward_joint with perturbed y (L_res on B_true = y_p - C) ==")
losses = model.forward_joint(y_p, c, boxes, (H, W), use_br=True, lambda_tebs=1.0,
                             lambda_res=1.0, gate_mode="learn")
for k, v in losses.items():
    if isinstance(v, torch.Tensor) and v.dim() == 0:
        print(f"  {k}: {v.item():.4f}")
assert "res" in losses and torch.isfinite(losses["res"])
print("res loss:", losses["res"].item())

print("== backward through gate + residual branch ==")
losses["total"].backward()
names = [n for n, p in model.named_parameters()
         if (("residual_branch" in n or "gate_head" in n)) and p.grad is not None
         and torch.isfinite(p.grad).all()]
print("residual/gate params with finite grad:", names)
assert names, "residual branch / gate received no gradient"

print("== forward_nuc_pretrain (gate fixed at 1) ==")
coef_gt = torch.rand(2, 10)
l = model.forward_nuc_pretrain(y_p, coef_gt, c=c, lambda_res=1.0)
print("  cor:", l["cor"].item(), "res:", l["res"].item())
l["total"].backward()

print("== state_dict round trip (residual enabled) ==")
sd = model.state_dict()
keys = [k for k in sd if "residual_branch" in k or "gate_head" in k]
print("residual/gate keys:", len(keys))
assert keys

print("== detect (inference) ==")
with torch.no_grad():
    res = model.detect(y, score_thr=0.05)
    print("  boxes:", res[0][0].shape)

print("SMOKE PR OK")
