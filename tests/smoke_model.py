# -*- coding: utf-8 -*-
"""Quick smoke test: model forward/backward/loss shape checks on random data.

Baseline reproduction configuration (residual NUC disabled).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from unied.models.unied import UniCD
from unied.models.head import build_targets, decode_predictions

torch.manual_seed(0)

print("== build model (baseline, no residual) ==")
model = UniCD(degree=3)
n_params = sum(p.numel() for p in model.parameters())
print(f"total params: {n_params / 1e6:.4f} M")

H, W = 512, 640
y = torch.rand(2, 3, H, W)
c = torch.rand(2, 3, H, W)
boxes = torch.tensor([[[100.0, 50, 130, 90], [0, 0, 0, 0]],
                      [[200.0, 300, 220, 310], [400, 100, 430, 150]]])
model.set_image_size(H, W)

print("== correct ==")
r = model.correct(y)
print("corrected shape:", r.shape, "range", float(r.min()), float(r.max()))

print("== forward_joint ==")
losses = model.forward_joint(y, c, boxes, (H, W), use_br=True, lambda_tebs=1.0)
for k, v in losses.items():
    print(f"  {k}: {v.item():.4f}")
assert all(torch.isfinite(v) for v in losses.values())

print("== backward ==")
losses["total"].backward()
grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
print("gradients finite:", grads_ok)

print("== forward_nuc_pretrain ==")
coef_gt = torch.rand(2, 10)
l = model.forward_nuc_pretrain(y, coef_gt)
print("  cor:", l["cor"].item())
l["total"].backward()

print("== forward_det_only ==")
l2 = model.forward_det_only(y, boxes, (H, W))
print("  det:", l2["det"].item())

print("== detect (inference) ==")
with torch.no_grad():
    res = model.detect(y, score_thr=0.05)
    print("  boxes:", res[0][0].shape, "scores:", res[0][1].shape)

print("== build_targets / decode round trip ==")
tgt = build_targets(boxes[0], (H, W))
print("  level sizes:", [t["cls"].shape for t in tgt])
print("SMOKE OK")
