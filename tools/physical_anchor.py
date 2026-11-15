# -*- coding: utf-8 -*-
"""
Physical-anchor analysis for the parameter-residual decomposition.

Two complementary, assumption-free checks using REAL paired data from
IRBFD-syn (no model inference, no fabricated sensor metadata):

 (1) Residual structure: for each degraded/clear pair, the true bias field is
     B = Y - C. Fit the best degree-3 polynomial bias field by least squares
     (the same 10-term basis used by the model); the unexplained part
     B_res = B - B_poly is the "non-polynomial residual". Measure how much of
     that residual energy is column-stripe energy (energy of the column-mean
     profile after removing its smooth component) vs. other structure. A high
     stripe fraction directly supports the paper's claim that the residual
     branch captures column-wise readout nonuniformity.

 (2) Polynomial coefficient statistics: distribution of the 10 fitted
     coefficients across the test set and their relation to scene mean
     intensity (a proxy for operating-point / focal-plane temperature change).

Outputs JSON + two PDF figures suitable for the paper.

    python tools/physical_anchor.py --data-root /root/work/UniCD \
        --out-dir /root/work/UniCD/UniCD-repro/results/physical_anchor
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def poly_basis(h, w, degree=3):
    """Same normalized polynomial basis as unied.poly (x,y in [-1,1])."""
    xs = np.linspace(-1, 1, w, dtype=np.float64)
    ys = np.linspace(-1, 1, h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    cols = []
    for t in range(degree + 1):
        for s in range(degree + 1 - t):
            cols.append((xx ** t) * (yy ** s))
    return np.stack(cols, axis=0)  # [K,H,W]


def analyse_pair(y, c, basis, pinv):
    b = y - c
    # least-squares polynomial fit to the TRUE bias field; pseudo-inverse of
    # the fixed basis is pre-computed once (10 x HW), making this a matvec.
    coef = pinv @ b.reshape(-1)
    b_poly = np.tensordot(coef, basis, axes=(0, 0))
    b_res = b - b_poly

    # column-stripe energy of the residual: energy in the column-mean profile
    col = b_res.mean(axis=0)
    stripe = col - gaussian_filter(col, sigma=12)
    e_res = float(np.mean(b_res ** 2)) + 1e-12
    e_stripe = float(np.mean(stripe ** 2))
    # fraction of residual variance explained by column-wise structure
    col_field = np.broadcast_to(col[np.newaxis, :], b_res.shape)
    e_col_field = float(np.mean(col_field ** 2))
    frac_col = e_col_field / e_res
    frac_stripe_hf = e_stripe / (float(np.mean(col ** 2)) + 1e-12)
    return {
        'coef': coef,
        'e_res': e_res,
        'e_poly': float(np.mean(b_poly ** 2)),
        'e_total': float(np.mean(b ** 2)),
        'frac_col': frac_col,
        'col_stripe_rms': float(np.std(stripe)),
        'res_rms': float(np.std(b_res)),
        'scene_mean': float(np.mean(c)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/root/work/UniCD')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out-dir',
                    default='/root/work/UniCD/UniCD-repro/results/physical_anchor')
    args = ap.parse_args()

    voc_root = Path(args.data_root) / 'IRBFD-syn' / 'VOC2007'
    stems = [l.strip() for l in
             (voc_root / 'ImageSets' / 'Main' / 'test.txt').read_text().splitlines()
             if l.strip()]
    if args.limit:
        stems = stems[:args.limit]
    print('test stems: %d' % len(stems), flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    basis = None
    pinv = None
    coefs, frac_cols, res_rms, stripe_rms, scene_means, poly_share = \
        [], [], [], [], [], []

    n = 0
    for stem in stems:
        yp = voc_root / 'JPEGImages-bias' / f'{stem}.jpg'
        cp = voc_root / 'JPEGImages-clear' / f'{stem}.jpg'
        if not (yp.exists() and cp.exists()):
            continue
        y = np.asarray(Image.open(yp).convert('L'), dtype=np.float64) / 255.0
        c = np.asarray(Image.open(cp).convert('L'), dtype=np.float64) / 255.0
        if basis is None:
            basis = poly_basis(*y.shape)
            flat = basis.reshape(basis.shape[0], -1).T
            pinv = np.linalg.pinv(flat)
        r = analyse_pair(y, c, basis, pinv)
        coefs.append(r['coef'])
        frac_cols.append(r['frac_col'])
        res_rms.append(r['res_rms'])
        stripe_rms.append(r['col_stripe_rms'])
        scene_means.append(r['scene_mean'])
        poly_share.append(r['e_poly'] / (r['e_total'] + 1e-12))
        n += 1
        if n % 500 == 0:
            print('processed %d' % n, flush=True)

    coefs = np.asarray(coefs)
    frac_cols = np.asarray(frac_cols)
    res_rms = np.asarray(res_rms)
    stripe_rms = np.asarray(stripe_rms)
    scene_means = np.asarray(scene_means)
    poly_share = np.asarray(poly_share)

    # Pearson correlation: residual RMS vs column-stripe RMS
    corr = float(np.corrcoef(res_rms, stripe_rms)[0, 1])
    # rank correlation (Spearman, computed without scipy.stats dependency)
    rr = np.argsort(np.argsort(res_rms)).astype(float)
    ss = np.argsort(np.argsort(stripe_rms)).astype(float)
    spearman = float(np.corrcoef(rr, ss)[0, 1])
    # coefficient drift vs scene mean (a0 = constant offset term)
    a0 = coefs[:, 0]
    corr_a0_scene = float(np.corrcoef(a0, scene_means)[0, 1])

    summary = {
        'n_images': n,
        'poly_energy_share_mean': float(np.mean(poly_share)),
        'poly_energy_share_median': float(np.median(poly_share)),
        'residual_column_variance_fraction_mean': float(np.mean(frac_cols)),
        'residual_column_variance_fraction_median': float(np.median(frac_cols)),
        'pearson_resRMS_vs_stripeRMS': corr,
        'spearman_resRMS_vs_stripeRMS': spearman,
        'pearson_a0_vs_scene_mean': corr_a0_scene,
        'coef_mean': coefs.mean(axis=0).tolist(),
        'coef_std': coefs.std(axis=0).tolist(),
        'coef_p05': np.percentile(coefs, 5, axis=0).tolist(),
        'coef_p95': np.percentile(coefs, 95, axis=0).tolist(),
    }
    print(json.dumps({k: v for k, v in summary.items()
                      if not isinstance(v, list)}, indent=2), flush=True)
    with open(out / 'physical_anchor.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Fig 1: coefficient distribution (box plot) ----
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = ['$a_{%d}$' % i for i in range(10)]
    ax.boxplot([coefs[:, i] for i in range(10)], labels=labels,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#9ecae1', color='#3182bd'),
               medianprops=dict(color='#c51b7d'))
    ax.set_ylabel('Fitted coefficient value')
    ax.set_xlabel('Degree-3 polynomial basis term')
    ax.set_title('Distribution of polynomial bias coefficients (IRBFD-syn test)')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out / 'coef_distribution.pdf')
    plt.close(fig)

    # ---- Fig 2: residual RMS vs column-stripe RMS ----
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    idx = np.random.RandomState(0).choice(n, size=min(n, 1200), replace=False)
    ax.scatter(stripe_rms[idx] * 255, res_rms[idx] * 255, s=6, alpha=0.35,
               color='#3182bd', edgecolors='none')
    ax.set_xlabel('Column-stripe RMS of residual (8-bit units)')
    ax.set_ylabel('Residual RMS (8-bit units)')
    ax.set_title('Residual energy vs. column-stripe strength\n'
                 r'$\rho=%.3f$, Spearman $=%.3f$' % (corr, spearman))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'residual_vs_stripe.pdf')
    plt.close(fig)

    # ---- Fig 3: a0 drift vs scene mean ----
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(scene_means[idx] * 255, a0[idx] * 255, s=6, alpha=0.35,
               color='#e6550d', edgecolors='none')
    ax.set_xlabel('Scene mean intensity (8-bit units)')
    ax.set_ylabel('Constant bias term $a_0$ (8-bit units)')
    ax.set_title('Polynomial offset vs. scene operating point\n'
                 r'$\rho=%.3f$' % corr_a0_scene)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'coef_drift.pdf')
    plt.close(fig)

    print('PHYSICAL_ANCHOR_DONE', flush=True)


if __name__ == '__main__':
    main()
