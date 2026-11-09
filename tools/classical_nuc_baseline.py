# -*- coding: utf-8 -*-
"""
Classical NUC baseline comparison for the paper.
Implements: Moment Matching (MM), Two-Point Correction (TPC), and No Correction.
Evaluates on IRBFD-syn test set with PSNR/SSIM/SCRG.

    python tools/classical_nuc_baseline.py --data-root /root/work/UniCD
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from unied.utils.metrics import psnr, ssim, scrg


def load_test_pairs(voc_root, limit=0):
    """Load degraded/clear pairs from IRBFD-syn test split."""
    split_file = Path(voc_root) / 'ImageSets' / 'Main' / 'test.txt'
    stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    if limit > 0:
        stems = stems[:limit]
    
    bias_dir = Path(voc_root) / 'JPEGImages-bias'
    clear_dir = Path(voc_root) / 'JPEGImages-clear'
    
    pairs = []
    for st in stems:
        y_path = bias_dir / f'{st}.jpg'
        c_path = clear_dir / f'{st}.jpg'
        if y_path.exists() and c_path.exists():
            y = np.asarray(Image.open(y_path).convert('L'), dtype=np.float32) / 255.0
            c = np.asarray(Image.open(c_path).convert('L'), dtype=np.float32) / 255.0
            pairs.append((st, y, c))
    return pairs


def moment_matching_correction(y_seq):
    """Moment Matching NUC: equalize row/column means and stds to global stats."""
    global_mean = y_seq.mean()
    global_std = y_seq.std() + 1e-8
    
    # Row-wise statistics
    row_means = y_seq.mean(axis=1, keepdims=True)
    row_stds = y_seq.std(axis=1, keepdims=True) + 1e-8
    
    # Column-wise statistics
    col_means = y_seq.mean(axis=0, keepdims=True)
    col_stds = y_seq.std(axis=0, keepdims=True) + 1e-8
    
    # Apply moment matching
    corrected = (y_seq - row_means) / row_stds * global_std + global_mean
    corrected = (corrected - col_means) / col_stds * global_std + global_mean
    corrected = 0.5 * corrected + 0.5 * y_seq  # blend
    
    return np.clip(corrected, 0, 1)


def two_point_correction(y, c_gt=None):
    """Oracle two-point correction: optimal affine Y -> a*Y + b."""
    if c_gt is None:
        return y
    Y_flat = y.flatten()
    C_flat = c_gt.flatten()
    A = np.stack([Y_flat, np.ones_like(Y_flat)], axis=1)
    result, _, _, _ = np.linalg.lstsq(A, C_flat, rcond=None)
    a, b = result
    corrected = a * y + b
    return np.clip(corrected, 0, 1)


def fixed_pattern_correction(y):
    """Remove low-frequency bias via Gaussian high-pass filtering."""
    from scipy.ndimage import gaussian_filter
    bias = gaussian_filter(y, sigma=30)
    corrected = y - bias + bias.mean()
    return np.clip(corrected, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/root/work/UniCD')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output', default='/root/work/UniCD/UniCD-repro/results/classical_nuc.json')
    args = ap.parse_args()
    
    voc_root = Path(args.data_root) / 'IRBFD-syn' / 'VOC2007'
    print(f'Loading test pairs from {voc_root}...')
    pairs = load_test_pairs(voc_root, limit=args.limit)
    print(f'Loaded {len(pairs)} test pairs')
    
    methods = {
        'no_correction': lambda y, c: y,
        'moment_matching': lambda y, c: moment_matching_correction(y),
        'two_point_oracle': lambda y, c: two_point_correction(y, c),
        'fixed_pattern_removal': lambda y, c: fixed_pattern_correction(y),
    }
    
    results = {}
    for name, fn in methods.items():
        psnr_vals, ssim_vals, scrg_vals = [], [], []
        for stem, y, c in pairs:
            r = fn(y, c)
            psnr_vals.append(psnr(c, r))
            ssim_vals.append(ssim(c, r))
            scrg_vals.append(scrg(y, r, c))
        
        results[name] = {
            'n_images': len(pairs),
            'PSNR': float(np.mean(psnr_vals)),
            'PSNR_std': float(np.std(psnr_vals)),
            'SSIM': float(np.mean(ssim_vals)),
            'SSIM_std': float(np.std(ssim_vals)),
            'SCRG': float(np.mean(scrg_vals)),
            'SCRG_std': float(np.std(scrg_vals)),
        }
        print(f'{name}: PSNR={results[name]["PSNR"]:.2f}±{results[name]["PSNR_std"]:.2f} '
              f'SSIM={results[name]["SSIM"]:.4f} SCRG={results[name]["SCRG"]:.3f}')
    
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
