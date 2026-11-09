# -*- coding: utf-8 -*-
"""
IRBFD dataset loaders (VOC2007 layout) for IRBFD-syn and IRBFD-real.

Layout (as shipped):
    IRBFD-syn/VOC2007: Annotations/ ImageSets/Main/{train,val,test}.txt
                       JPEGImages-bias/{stem}.jpg   (degraded Y)
                       JPEGImages-clear/{stem}.jpg  (clear C)
    IRBFD-real/VOC2007: Annotations/ ImageSets/Main/{train,val,test}.txt
                       JPEGImages/{stem}.jpg        (real degraded Y)

All images are 640x512, single class 'UAV'. Annotations are VOC XML.
NOTE: some real XML files advertise a ".png" filename while the actual files
are ".jpg" -- matching is done by stem, not by the XML <filename> field.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .coef_fit import load_coef_cache

CLASSES = ["UAV"]


def _parse_boxes(xml_path: Path) -> np.ndarray:
    """Return [N,4] xyxy float boxes in absolute pixel coordinates."""
    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue
        bb = obj.find("bndbox")
        x1 = float(bb.findtext("xmin"))
        y1 = float(bb.findtext("ymin"))
        x2 = float(bb.findtext("xmax"))
        y2 = float(bb.findtext("ymax"))
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class IRBFDDataset(Dataset):
    def __init__(
        self,
        voc_root: str,
        split: str = "train",
        mode: str = "syn",            # 'syn' or 'real'
        coef_cache=None,              # .npz produced by prepare_coef_cache
        scale: float = 1.0,           # resize factor (1.0 keeps 640x512)
        hflip: float = 0.0,           # random horizontal flip probability
    ):
        assert mode in ("syn", "real")
        self.root = Path(voc_root)
        self.split = split
        self.mode = mode
        self.scale = scale
        self.hflip = hflip

        txt = self.root / "ImageSets" / "Main" / f"{split}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"missing image set: {txt}")
        # NOTE: the real ImageSets entries carry a ".jpg" extension while the
        # synthetic ones do not; strip any extension so both are matched by stem.
        self.stems = [Path(ln.strip()).stem for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        # keep only entries that actually have images
        self.stems = [s for s in self.stems if self._image_path(s).exists()]

        if mode == "syn" and coef_cache is not None:
            cache = load_coef_cache(coef_cache)
            self.coefs = {s: c for s, c in cache.items() if s in set(self.stems)}
            # keep only stems that have a cached coefficient target
            self.stems = [s for s in self.stems if s in self.coefs]
        else:
            self.coefs = None

    # -- paths -------------------------------------------------------------
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def _image_path(self, stem: str) -> Path:
        folder = "JPEGImages-bias" if self.mode == "syn" else "JPEGImages"
        base = self.root / folder / stem
        for ext in self.IMAGE_EXTS:
            p = Path(str(base) + ext)
            if p.exists():
                return p
        # fall back to .jpg so callers get a stable path for error messages
        return Path(str(base) + ".jpg")

    def _clear_path(self, stem: str) -> Path:
        base = self.root / "JPEGImages-clear" / stem
        for ext in self.IMAGE_EXTS:
            p = Path(str(base) + ext)
            if p.exists():
                return p
        return Path(str(base) + ".jpg")

    def _xml_path(self, stem: str) -> Path:
        base = self.root / "Annotations" / stem
        for ext in (".xml",):
            p = Path(str(base) + ext)
            if p.exists():
                return p
        return Path(str(base) + ".xml")

    # -- dataset protocol ----------------------------------------------------
    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        y = self._load(self._image_path(stem))          # [3,H,W] float [0,1]
        c = self._load(self._clear_path(stem)) if self.mode == "syn" else None
        boxes = _parse_boxes(self._xml_path(stem))      # [N,4] xyxy

        h, w = y.shape[1], y.shape[2]
        if self.scale != 1.0:
            nh, nw = int(round(h * self.scale)), int(round(w * self.scale))
            y = torch.nn.functional.interpolate(y.unsqueeze(0), (nh, nw), mode="bilinear", align_corners=False).squeeze(0)
            if c is not None:
                c = torch.nn.functional.interpolate(c.unsqueeze(0), (nh, nw), mode="bilinear", align_corners=False).squeeze(0)
            boxes[:, [0, 2]] *= nw / w
            boxes[:, [1, 3]] *= nh / h

        if self.hflip > 0 and np.random.rand() < self.hflip:
            y = torch.flip(y, dims=[2])
            if c is not None:
                c = torch.flip(c, dims=[2])
            new_x1 = w - boxes[:, 2]
            new_x2 = w - boxes[:, 0]
            boxes[:, 0], boxes[:, 2] = new_x1, new_x2

        item = {
            "y": y,
            "c": c,
            "boxes": torch.from_numpy(boxes).float(),
            "stem": stem,
            "n_boxes": boxes.shape[0],
        }
        if self.coefs is not None:
            item["coef_gt"] = torch.from_numpy(self.coefs[stem]).float()
        return item

    def _load(self, path: Path) -> torch.Tensor:
        from PIL import Image

        # Infrared images are intrinsically single-channel; the stored JPEGs are
        # (near-)replicated RGB. We load as grayscale and replicate to 3 channels
        # so that the predicted scalar bias field applies consistently to every
        # channel (and matches the coefficient fitting in coef_fit.py).
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0  # [H,W]
        arr = np.stack([arr] * 3, axis=-1)
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def collate_irbfd(batch):
    """Pad boxes to the max box count within the batch."""
    y = torch.stack([b["y"] for b in batch])
    has_c = all(b["c"] is not None for b in batch)
    c = torch.stack([b["c"] for b in batch]) if has_c else None
    max_n = max(b["n_boxes"] for b in batch)
    boxes = torch.zeros((len(batch), max_n, 4), dtype=torch.float32)
    for i, b in enumerate(batch):
        if b["n_boxes"] > 0:
            boxes[i, : b["n_boxes"]] = b["boxes"]
    out = {
        "y": y,
        "c": c,
        "boxes": boxes,
        "stems": [b["stem"] for b in batch],
        "n_boxes": [b["n_boxes"] for b in batch],
    }
    if "coef_gt" in batch[0]:
        out["coef_gt"] = torch.stack([b["coef_gt"] for b in batch])
    return out
