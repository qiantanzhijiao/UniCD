# -*- coding: utf-8 -*-
"""
Ground-truth polynomial coefficient fitting for IRBFD-syn.

The paper trains the NUC module with the MAE loss (Eq. 5)

    L_cor = (1/N) * |a_hat - a|_1

where `a` are "predefined coefficients" of the bias field. The bias field
generation parameters are not released with the dataset, so we recover a
per-image coefficient vector `a` by least-squares fitting of the degree-3
bivariate polynomial to the observed bias field  B = Y - C  (C is the paired
clear image provided by IRBFD-syn). The fit is performed on a strided grid,
which is numerically accurate for smooth low-frequency bias fields and fast
enough to run once over all 30,000 images (result cached as an .npz file).

The monomial order matches `unied.poly.monomial_basis`.
"""
import os
from pathlib import Path

import numpy as np
import torch

from .. import poly

COEF_STEP = 4  # strided sampling for the least-squares fit


def numpy_basis(height: int, width: int, degree: int = poly.DEFAULT_DEGREE) -> np.ndarray:
    """[K, H, W] monomial basis on a [0,1] grid normalised to [-1,1]."""
    return poly.monomial_basis(height, width, degree, normalized=True).numpy()


def fit_polynomial_coefs(
    bias: np.ndarray,
    basis: np.ndarray,
    step: int = COEF_STEP,
) -> np.ndarray:
    """Fit degree-D polynomial coefficients to a bias field image.

    bias: [H, W] float array; basis: [K, H, W]; returns coefs [K].
    """
    h, w = bias.shape
    k = basis.shape[0]
    # design matrix over the strided grid: A = [basis_k(y,x)]_pixels x K
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    A = basis[:, ys][:, :, xs].reshape(k, -1).T  # [P, K]
    b = bias[ys][:, xs].reshape(-1)               # [P]
    coefs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coefs.astype(np.float32)


def prepare_coef_cache(
    syn_voc_root: str,
    out_path: str,
    split: str = "trainval",
    step: int = COEF_STEP,
) -> None:
    """
    Pre-compute the polynomial coefficient targets for all IRBFD-syn images
    listed in ImageSets/Main/{split}.txt and store them in out_path (.npz).

    The .npz holds: 'stems' (S,), 'coefs' (S, K).
    """
    root = Path(syn_voc_root)
    main = root / "ImageSets" / "Main"
    txt = main / f"{split}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"missing image set file: {txt}")
    stems = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    stems = [s for s in stems if (root / "JPEGImages-clear" / f"{s}.jpg").exists()]

    img_h, img_w = 512, 640
    basis = numpy_basis(img_h, img_w)
    coefs = np.zeros((len(stems), basis.shape[0]), dtype=np.float32)
    seen = set()
    for i, stem in enumerate(stems):
        if stem in seen:
            continue
        seen.add(stem)
        y = _read_gray(root / "JPEGImages-bias" / f"{stem}.jpg")
        c = _read_gray(root / "JPEGImages-clear" / f"{stem}.jpg")
        b = y.astype(np.float32) - c.astype(np.float32)
        coefs[i] = fit_polynomial_coefs(b, basis, step=step)
        if (i + 1) % 2000 == 0:
            print(f"[prepare_coef_cache] {i + 1}/{len(stems)} done")
    np.savez(out_path, stems=np.array(stems, dtype=object), coefs=coefs)
    print(f"[prepare_coef_cache] saved {len(stems)} coefficient targets -> {out_path}")


def _read_gray(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    return arr


def load_coef_cache(cache_path: str):
    """Load the coefficient cache; returns {stem: coefs[K]}."""
    data = np.load(cache_path, allow_pickle=True)
    stems = [str(s) for s in data["stems"]]
    # Normalize coefficients from 8-bit image scale to [0,1] scale.
    # The cache was built from B = Y - C in [0,255]; training images are in [0,1].
    coefs = data["coefs"] / 255.0
    return {st: coefs[i] for i, st in enumerate(stems)}
