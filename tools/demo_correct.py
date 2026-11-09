# -*- coding: utf-8 -*-
"""
Correct arbitrary infrared images with a trained NUC module and save the
corrected images + estimated bias fields.

    python tools/demo_correct.py --ckpt outputs/epoch_050.pt \
        --input path/to/image_or_folder --out outputs/demo

The correction follows R = Y - B(Y) with the degree-3 polynomial bias field,
or the parametric + residual decomposition when the checkpoint was trained
with the residual branch (pass the matching --config).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from unied.models.unied import UniCD


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--config", default=None,
                    help="optional config (e.g. configs_improved/eval_pr.yaml) so a "
                         "checkpoint trained with the residual branch loads correctly")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    residual = None
    if args.config:
        from unied import load_config

        cfg = load_config(args.config)
        residual = cfg.model.get("residual")
    model = UniCD(degree=3, residual=residual)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    src = Path(args.input)
    imgs = [src] if src.is_file() else sorted(src.glob("*.*"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for p in imgs:
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            y = load_image(p).to(device)
            r = model.correct(y).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
            b = model.bias_field(y)[0].cpu().numpy().transpose(1, 2, 0)
            b = b - b.min()
            b = b / (b.max() + 1e-6)
            stem = p.stem
            Image.fromarray((np.clip(r, 0, 1) * 255).astype(np.uint8)).save(
                out_dir / f"{stem}_corrected.png"
            )
            Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8)).save(
                out_dir / f"{stem}_bias.png"
            )
            print(f"[demo] {p.name} -> {out_dir / (stem + '_corrected.png')}")


if __name__ == "__main__":
    main()
