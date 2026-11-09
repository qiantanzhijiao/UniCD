# -*- coding: utf-8 -*-
"""
Prepare the ground-truth polynomial coefficient targets for IRBFD-syn.

    python tools/prepare_coefs.py --syn-root /root/work/UniCD/IRBFD-syn/VOC2007 \\
        --out outputs/coef_cache_syn.npz --split trainval

Run once; the cached .npz is then used by the NUC pre-training stage.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unied.datasets.coef_fit import prepare_coef_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn-root", required=True)
    ap.add_argument("--out", default="outputs/coef_cache_syn.npz")
    ap.add_argument("--split", default="trainval")
    args = ap.parse_args()
    prepare_coef_cache(args.syn_root, args.out, split=args.split)


if __name__ == "__main__":
    main()
