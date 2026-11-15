# -*- coding: utf-8 -*-
"""
SIRST cross-dataset loader (IRSTD-1K / NUAA-SIRST / NUDT-SIRST).

Layout (under one root):
    IRSTD-1K/ images/ masks/ img_idx/{train,test}_IRSTD-1K.txt
    NUAA-SIRST/ images/ masks/ img_idx/{train,test}_NUAA-SIRST.txt
    NUDT-SIRST/ images/ masks/ img_idx/{train,test}_NUD T-SIRST.txt  (no space)

Design:
    SIRST images are clean small-target frames. To exercise the NUC pipeline
    we synthesize a polynomial nonuniformity with coefficients sampled from the
    empirical IRBFD-syn coefficient distribution, i.e.
        Y = clamp(C + B(coefs) , 0, 1),   B = basis @ (coefs / 255)
    where the coef distribution is loaded from coef_cache_syn.npz so the
    degradation statistics match the training domain exactly.
    Detection targets are derived from the pixel masks (connected components
    -> boxes), keeping the same item format as IRBFDDataset so collate_irbfd
    can be reused unchanged.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from unied import poly

DATASETS = {
    "IRSTD-1K": ("IRSTD-1K", "train_IRSTD-1K.txt", "test_IRSTD-1K.txt"),
    "NUAA-SIRST": ("NUAA-SIRST", "train_NUAA-SIRST.txt", "test_NUAA-SIRST.txt"),
    "NUDT-SIRST": ("NUDT-SIRST", "train_NUDT-SIRST.txt", "test_NUDT-SIRST.txt"),
}


def _mask_to_boxes(mask: np.ndarray, min_px: int = 2) -> np.ndarray:
    """Connected components of a binary mask -> [N,4] xyxy boxes."""
    binm = (mask > 0).astype(np.uint8)
    if _HAS_CV2:
        n, _, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_px:
                continue
            boxes.append([x, y, x + w - 1, y + h - 1])
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    # fallback: single box around all foreground pixels
    ys, xs = np.nonzero(binm)
    if len(ys) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)


class SIRSTDataset(Dataset):
    """
    Returns the same item dict as IRBFDDataset:
        y [3,H,W] degraded, c [3,H,W] clean, boxes [N,4], coef_gt [K],
        stem, n_boxes
    """

    def __init__(
        self,
        root: str,
        dataset: str = "IRSTD-1K",
        split: str = "train",
        coef_pool=None,            # np.ndarray [M, K] empirical coefs (0-255 scale)
        size: int = 512,           # square resize target
        hflip: float = 0.5,
        stripe_p: float = 0.3,     # chance of adding column-stripe noise
        seed: int = 42,
    ):
        assert dataset in DATASETS, f"unknown dataset {dataset}"
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.size = size
        self.hflip = hflip if split == "train" else 0.0
        self.stripe_p = stripe_p if split == "train" else 0.0
        self.rng = random.Random(seed)

        dirname, train_txt, test_txt = DATASETS[dataset]
        base = self.root / dirname
        idx_file = base / "img_idx" / (train_txt if split == "train" else test_txt)
        names = [l.strip() for l in idx_file.read_text().splitlines() if l.strip()]
        self.names = []
        for nm in names:
            p = base / "images" / nm
            if not p.exists():
                cand = list((base / "images").glob(nm + ".*"))
                if cand:
                    p = cand[0]
                else:
                    continue
            self.names.append(p.name)

        self.img_dir = base / "images"
        self.mask_dir = base / "masks"
        self.degree = poly.DEFAULT_DEGREE
        self.n_coefs = poly.num_coefficients(self.degree)
        if coef_pool is not None:
            self.coef_pool = np.asarray(coef_pool, dtype=np.float32)
        else:
            # neutral pool: zero bias
            self.coef_pool = np.zeros((1, self.n_coefs), dtype=np.float32)

    def __len__(self):
        return len(self.names)

    def _load_gray(self, path: Path) -> np.ndarray:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32) / 255.0  # [H,W]

    def _resize(self, arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
        s = self.size
        if _HAS_CV2:
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
            return cv2.resize(arr, (s, s), interpolation=interp)
        from PIL import Image
        im = Image.fromarray((arr * 255).astype(np.uint8) if not is_mask else arr.astype(np.uint8))
        im = im.resize((s, s), Image.NEAREST if is_mask else Image.BILINEAR)
        out = np.asarray(im, dtype=np.float32)
        return out / 255.0 if not is_mask else out

    def __getitem__(self, idx: int):
        name = self.names[idx]
        c = self._load_gray(self.img_dir / name)                 # clean
        m = self._load_gray(self.mask_dir / name)                # mask 0/1

        c = self._resize(c, is_mask=False)
        m = self._resize(m, is_mask=True)

        h = w = self.size
        basis_np = poly.monomial_basis(h, w, self.degree).numpy()   # [K,H,W]

        # --- synthesize nonuniformity from empirical IRBFD distribution ---
        coefs = self.coef_pool[np.random.randint(len(self.coef_pool))].copy()
        bias = np.einsum("k,khw->hw", coefs / 255.0, basis_np)      # [H,W]
        y = np.clip(c + bias, 0.0, 1.0)

        if self.stripe_p > 0 and self.rng.random() < self.stripe_p:
            n_stripe = self.rng.randint(3, 12)
            cols = self.rng.sample(range(w), n_stripe)
            amp = self.rng.uniform(0.02, 0.08)
            for cx in cols:
                y[:, cx] = np.clip(y[:, cx] + self.rng.choice([-1, 1]) * amp, 0, 1)

        boxes = _mask_to_boxes(m)

        if self.hflip > 0 and self.rng.random() < self.hflip:
            y = np.ascontiguousarray(y[:, ::-1])
            c = np.ascontiguousarray(c[:, ::-1])
            m = np.ascontiguousarray(m[:, ::-1])
            if boxes.shape[0] > 0:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - 1 - x2
                boxes[:, 2] = w - 1 - x1

        # grayscale -> 3-channel (match IRBFD convention)
        y3 = np.stack([y] * 3, axis=0)
        c3 = np.stack([c] * 3, axis=0)

        return {
            "y": torch.from_numpy(y3).float(),
            "c": torch.from_numpy(c3).float(),
            "boxes": torch.from_numpy(boxes).float(),
            "coef_gt": torch.from_numpy(coefs / 255.0).float(),
            "stem": name,
            "n_boxes": int(boxes.shape[0]),
        }


def load_coef_pool(npz_path: str) -> np.ndarray:
    """Load empirical coefficient matrix [M, K] from coef_cache_syn.npz."""
    d = np.load(npz_path, allow_pickle=True)
    return np.asarray(d["coefs"], dtype=np.float32)
