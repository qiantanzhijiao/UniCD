# -*- coding: utf-8 -*-
"""
Verify the GT coefficient cache: refit a few stems and compare with the cache.
Data root from UNIED_ROOT (default Windows E:/work/UniCD).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from unied.datasets.coef_fit import numpy_basis, fit_polynomial_coefs, load_coef_cache

ROOT = Path(os.environ.get("UNIED_ROOT", r"E:/work/UniCD"))

syn = ROOT / "IRBFD-syn" / "VOC2007"
cache = load_coef_cache(str(ROOT / "UniCD-repro" / "outputs" / "coef_cache_syn.npz"))
basis = numpy_basis(512, 640)

stems = list(cache)[:3]
for stem in stems:
    with Image.open(syn / "JPEGImages-bias" / f"{stem}.jpg") as im:
        y = np.asarray(im.convert("L"), np.float32)
    with Image.open(syn / "JPEGImages-clear" / f"{stem}.jpg") as im:
        c = np.asarray(im.convert("L"), np.float32)
    b_true = y - c
    a_fit = fit_polynomial_coefs(b_true, basis, step=4)
    b_recon = np.einsum("k,kHW->HW", a_fit, basis)
    # compare cache coefs to refit coefs
    print(f"{stem}: |cache-refit| = {np.abs(cache[stem]-a_fit).max():.2e}",
          f"bias range=[{b_true.min():.0f},{b_true.max():.0f}]",
          f"recon err (RMSE)={np.sqrt(np.mean((b_true-b_recon)**2)):.2f}",
          "coef norm=", round(float(np.linalg.norm(cache[stem])), 2))
