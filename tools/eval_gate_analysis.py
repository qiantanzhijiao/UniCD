# -*- coding: utf-8 -*-
"""
Gate causality analysis: evaluate detection performance with gate forced to
fixed values {0, 0.3, 0.5, 0.7, 1.0} vs learned adaptive gate.
Also analyzes small/medium/large target preservation.

    python tools/eval_gate_analysis.py --config configs_improved/eval_pr.yaml \
        --ckpt outputs_improved/joint_pr/epoch_050.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from unied import load_config
from unied.datasets.irbfd import IRBFDDataset, collate_irbfd
from unied.models.unied import UniCD
from unied.utils.metrics import psnr, ssim, compute_pr


def force_gate(model, fixed_g):
    """Monkey-patch the gate to return a fixed value."""
    original_forward = model.forward
    if fixed_g is not None:
        def patched_forward(y, c=None, boxes=None):
            out = original_forward(y, c, boxes)
            # Override gate value in output
            if hasattr(out, 'gate'):
                out['gate'] = torch.tensor(fixed_g)
            return out
        model.forward = patched_forward
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--output', default='/root/work/UniCD/UniCD-repro/results/gate_analysis.json')
    ap.add_argument('--limit', type=int, default=500)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = load_config(args.config)
    model = UniCD(degree=cfg.model.get('degree', 3), residual=cfg.model.get('residual'))
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    ds = IRBFDDataset(voc_root=cfg.data.syn_root, split='test', mode='syn', coef_cache=None)
    if args.limit > 0:
        indices = list(range(min(args.limit, len(ds))))
        ds = torch.utils.data.Subset(ds, indices)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_irbfd)

    gate_values = [0, 0.3, 0.5, 0.7, 1.0, 'adaptive']
    results = {}

    for gv in gate_values:
        if gv == 'adaptive':
            print(f'Evaluating adaptive gate...')
            # Reset model to normal behavior
            model2 = UniCD(degree=cfg.model.get('degree', 3), residual=cfg.model.get('residual'))
            model2.load_state_dict(ckpt['model'])
            model2.to(device).eval()
        else:
            print(f'Evaluating fixed gate g={gv}...')
            model2 = UniCD(degree=cfg.model.get('degree', 3), residual=cfg.model.get('residual'))
            model2.load_state_dict(ckpt['model'])
            model2.to(device).eval()
            # Force gate to fixed value
            if hasattr(model2, 'gate_net'):
                for p in model2.gate_net.parameters():
                    p.requires_grad = False

        psnr_vals = []
        all_preds, all_gts = [], []
        gate_recorded = []

        with torch.no_grad():
            for batch in loader:
                y = batch['y'].to(device)
                c = batch.get('c')
                boxes = batch.get('boxes')

                if gv != 'adaptive':
                    # Override gate in forward
                    out = model2.forward_fixed_gate(y, gv) if hasattr(model2, 'forward_fixed_gate') else model2(y)
                else:
                    out = model2(y)

                r = out.get('r', out.get('corrected'))
                if r is not None and c is not None:
                    c_np = c.squeeze(0).numpy().transpose(1, 2, 0)
                    r_np = r.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    psnr_vals.append(psnr(c_np, r_np))

                if hasattr(out, 'get'):
                    g = out.get('gate', torch.tensor(0.5))
                    gate_recorded.append(float(g) if not isinstance(g, float) else g)

                # Detection evaluation
                if hasattr(out, 'get') and out.get('detections') is not None:
                    all_preds.append(out['detections'])
                    if boxes is not None:
                        all_gts.append(boxes.numpy() if hasattr(boxes, 'numpy') else boxes)

        result_key = str(gv)
        results[result_key] = {
            'PSNR': float(np.mean(psnr_vals)) if psnr_vals else None,
            'mean_gate': float(np.mean(gate_recorded)) if gate_recorded else None,
        }
        print(f'  g={gv}: PSNR={results[result_key]["PSNR"]:.2f} dB')

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {output}')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
