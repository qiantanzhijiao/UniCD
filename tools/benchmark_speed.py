# -*- coding: utf-8 -*-
"""
Benchmark inference speed and model complexity for the paper.
Measures: parameters, FLOPs, latency, FPS.

    python tools/benchmark_speed.py --config configs/joint_train.yaml \
        --ckpt outputs/joint/epoch_050.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from unied import load_config
from unied.models.unied import UniCD


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, input_shape):
    """Rough FLOPs estimation via forward hook."""
    flops = [0]
    
    def hook_fn(module, input, output):
        if isinstance(module, torch.nn.Conv2d):
            batch_size = input[0].shape[0]
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            flops[0] += batch_size * out_h * out_w * kernel_flops * module.out_channels
        elif isinstance(module, torch.nn.Linear):
            flops[0] += module.in_features * module.out_features
    
    hooks = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))
    
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        model.nuc(dummy)
    
    for h in hooks:
        h.remove()
    
    return flops[0]


def benchmark_latency(model, input_shape, device, n_warmup=10, n_runs=100):
    model.eval()
    dev = next(model.parameters()).device
    dummy = torch.randn(*input_shape).to(dev)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model.nuc(dummy)
    
    if dev == 'cuda' or dev.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model.nuc(dummy)
            if dev == 'cuda' or dev.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times) * 1000  # ms
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'fps': float(1000.0 / np.mean(times)),
    }


def benchmark_full_pipeline(model, detector, input_shape, device, n_runs=50):
    """Benchmark full NUC + detection pipeline."""
    model.eval()
    if detector is not None:
        detector.eval()
    
    dummy = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            r = model.correct(dummy)
            if detector is not None:
                detector(r)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            r = model.correct(dummy)
            if detector is not None:
                detector(r)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times) * 1000
    return {
        'mean_ms': float(np.mean(times)),
        'fps': float(1000.0 / np.mean(times)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--output', default='/root/work/UniCD/UniCD-repro/results/speed_benchmark.json')
    args = ap.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    cfg = load_config(args.config)
    
    # Model variants to benchmark
    models = {}
    
    # 1. Baseline (poly only)
    model_base = UniCD(degree=3, residual=None)
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_base.load_state_dict(ckpt['model'], strict=False)
    except:
        pass
    model_base.to(device)
    models['baseline_poly'] = model_base
    
    # 2. With residual branch
    model_res = UniCD(degree=3, residual={'hidden': 16, 'layers': 3})
    model_res.to(device)
    models['with_residual'] = model_res
    
    input_shape = (1, 3, 512, 640)  # batch=1, channel=1, H=512, W=640
    
    results = {
        'device': device,
        'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
        'input_shape': list(input_shape),
    }
    
    for name, model in models.items():
        total, trainable = count_parameters(model)
        flops = estimate_flops(model, input_shape)
        latency = benchmark_latency(model, input_shape, device)
        
        results[name] = {
            'total_params': total,
            'trainable_params': trainable,
            'params_M': round(total / 1e6, 4),
            'flops_MFLOPs': round(flops / 1e6, 2),
            'latency_ms': latency['mean_ms'],
            'latency_std_ms': latency['std_ms'],
            'fps': latency['fps'],
        }
        
        print(f'\n{name}:')
        print(f'  Params: {total:,} ({total/1e6:.4f}M)')
        print(f'  FLOPs: {flops/1e6:.2f} MFLOPs')
        print(f'  Latency: {latency["mean_ms"]:.2f} ± {latency["std_ms"]:.2f} ms')
        print(f'  FPS: {latency["fps"]:.1f}')
    
    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {output}')


if __name__ == '__main__':
    main()
