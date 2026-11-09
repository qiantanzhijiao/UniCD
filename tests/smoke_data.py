# -*- coding: utf-8 -*-
"""
Dataset loader smoke test. The data root is taken from the UNIED_ROOT
environment variable if set, otherwise the Windows default is used:
    Windows default : E:/work/UniCD
    Linux example   : UNIED_ROOT=/root/work/UniCD python tests/smoke_data.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from unied.datasets.irbfd import IRBFDDataset, collate_irbfd
from unied.models.unied import UniCD

ROOT = Path(os.environ.get("UNIED_ROOT", r"E:/work/UniCD"))

print("== real dataset loader ==")
ds = IRBFDDataset(ROOT / "IRBFD-real" / "VOC2007", split="train", mode="real", hflip=0.5)
print("real train size:", len(ds))
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_irbfd)
b = next(iter(dl))
print("y:", b["y"].shape, "boxes:", b["boxes"].shape, "n_boxes:", b["n_boxes"])
print("stem:", b["stems"][:2])

print("== syn dataset loader (without coef cache) ==")
ds2 = IRBFDDataset(ROOT / "IRBFD-syn" / "VOC2007", split="test", mode="syn", hflip=0.0)
print("syn test size:", len(ds2))
dl2 = DataLoader(ds2, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_irbfd)
b2 = next(iter(dl2))
print("y:", b2["y"].shape, "c:", b2["c"].shape, "boxes:", b2["boxes"].shape)

print("== forward_det_only on real batch ==")
model = UniCD(degree=3)
losses = model.forward_det_only(b["y"], b["boxes"], (b["y"].shape[2], b["y"].shape[3]))
print("det:", losses["det"].item())

print("== detect on real batch ==")
with torch.no_grad():
    res = model.detect(b["y"], score_thr=0.5)
    print("n boxes:", res[0][0].shape)
print("DATA LOADER OK")
