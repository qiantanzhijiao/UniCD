# -*- coding: utf-8 -*-
"""
Unified, training-free NUC baseline evaluation on the IRBFD-syn test split.

This script replaces tools/classical_nuc_baseline.py and
tools/frequency_nuc_baseline.py for the paper table.  All methods are evaluated
on the SAME image pairs, with the SAME PSNR/SSIM/SCRG implementations, and SCRG
now uses the real VOC ground-truth bounding boxes read from Annotations/*.xml.

The earlier scripts passed the clean image array as the `boxes` argument of
scrg(), which produced ~8e4 degenerate pseudo-boxes per image and meaningless
SCRG values (plus a very large runtime).  This version fixes that.

Methods
  no_correction          : identity
  moment_matching        : row/column mean+std equalisation (Perry & Dereniak)
  two_point_oracle       : per-image least-squares affine fit Y -> aY+b to C
  fixed_pattern_removal  : Gaussian high-pass removal of low-frequency bias
  freq_notch             : Tendero-style Fourier notch destriping
  col_median             : column-median destriping

Usage:
  python tools/nuc_baselines_v2.py --data-root /root/work/UniCD \
      --out results/nuc_baselines_v2.json
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from unied.utils.metrics import psnr, ssim, scrg


METHODS = ("no_correction", "moment_matching", "two_point_oracle",
           "fixed_pattern_removal", "freq_notch", "col_median")


def load_test_stems(voc_root, limit=0):
    split_file = Path(voc_root) / "ImageSets" / "Main" / "test.txt"
    stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    if limit > 0 and limit < len(stems):
        idx = np.linspace(0, len(stems) - 1, limit).round().astype(int)
        stems = [stems[i] for i in sorted(set(int(i) for i in idx))]
    return stems


def load_pair(voc_root, stem):
    y_path = Path(voc_root) / "JPEGImages-bias" / f"{stem}.jpg"
    c_path = Path(voc_root) / "JPEGImages-clear" / f"{stem}.jpg"
    x_path = Path(voc_root) / "Annotations" / f"{stem}.xml"
    if not (y_path.exists() and c_path.exists()):
        return None
    y = np.asarray(Image.open(y_path).convert("L"), dtype=np.float32) / 255.0
    c = np.asarray(Image.open(c_path).convert("L"), dtype=np.float32) / 255.0
    boxes = []
    if x_path.exists():
        root = ET.parse(str(x_path)).getroot()
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append([float(bb.find("xmin").text), float(bb.find("ymin").text),
                          float(bb.find("xmax").text), float(bb.find("ymax").text)])
    return y, c, np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def moment_matching(y):
    gm, gs = y.mean(), y.std() + 1e-8
    rm, rs = y.mean(axis=1, keepdims=True), y.std(axis=1, keepdims=True) + 1e-8
    cm, cs = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True) + 1e-8
    a = (y - rm) / rs * gs + gm
    b = (a - cm) / cs * gs + gm
    return np.clip(0.5 * b + 0.5 * y, 0, 1)


def two_point_oracle(y, c):
    A = np.stack([y.ravel(), np.ones(y.size, dtype=np.float64)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, c.ravel().astype(np.float64), rcond=None)
    return np.clip(coef[0] * y + coef[1], 0, 1)


def fixed_pattern_removal(y):
    bias = gaussian_filter(y, sigma=30)
    return np.clip(y - bias + bias.mean(), 0, 1)


def freq_notch(y, notch_width=2, k_top=6, strength=0.85):
    f = np.fft.fft2(y)
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cx = w // 2
    profile = mag.mean(axis=0).copy()
    profile[cx - 3:cx + 4] = 0.0
    idx = np.argsort(profile)[-k_top:]
    mask = np.ones_like(profile, dtype=np.float32)
    for i in idx:
        lo, hi = max(0, i - notch_width), min(w, i + notch_width + 1)
        mask[lo:hi] = 1.0 - strength
    fshift = np.fft.fftshift(f)
    fshift *= mask[np.newaxis, :]
    r = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
    r = r - r.mean() + y.mean()
    return np.clip(r, 0, 1)


def col_median(y, sigma=12):
    col = np.median(y, axis=0)
    stripe = col - gaussian_filter(col, sigma=sigma)
    return np.clip(y - stripe[np.newaxis, :], 0, 1)


def correct_all(y, c):
    return {
        "no_correction": y,
        "moment_matching": moment_matching(y),
        "two_point_oracle": two_point_oracle(y, c),
        "fixed_pattern_removal": fixed_pattern_removal(y),
        "freq_notch": freq_notch(y),
        "col_median": col_median(y),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/root/work/UniCD")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out",
                    default="/root/work/UniCD/UniCD-repro/results/nuc_baselines_v2.json")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    voc_root = Path(args.data_root) / "IRBFD-syn" / "VOC2007"
    stems = load_test_stems(voc_root, limit=args.limit)
    print("test stems: %d" % len(stems), flush=True)

    acc = {m: {"psnr": [], "ssim": [], "scrg": []} for m in METHODS}
    n = 0
    for stem in stems:
        pair = load_pair(voc_root, stem)
        if pair is None:
            continue
        y, c, boxes = pair
        corrected = correct_all(y, c)
        for m in METHODS:
            r = corrected[m]
            acc[m]["psnr"].append(psnr(c, r))
            acc[m]["ssim"].append(ssim(c, r))
            # Only images with at least one valid GT box contribute to SCRG.
            if len(boxes) > 0:
                acc[m]["scrg"].append(scrg(y, r, boxes))
        n += 1
        if n % args.log_every == 0:
            print("processed %d" % n, flush=True)

    results = {"n_images": n, "scrg_images":
               {m: len(acc[m]["scrg"]) for m in METHODS}}
    for m in METHODS:
        a = acc[m]
        results[m] = {
            "PSNR": float(np.mean(a["psnr"])),
            "PSNR_std": float(np.std(a["psnr"])),
            "SSIM": float(np.mean(a["ssim"])),
            "SSIM_std": float(np.std(a["ssim"])),
            "SCRG": float(np.mean(a["scrg"])) if a["scrg"] else None,
            "SCRG_median": float(np.median(a["scrg"])) if a["scrg"] else None,
        }
        print("%s: %s" % (m, json.dumps(results[m])), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("NUC_BASELINES_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
