#!/usr/bin/env bash
# Improved method full pipeline + classical NUC + analysis
# Runs on server 192.168.0.4 (RTX 3060)
set -euo pipefail
cd /home/jgxy/work/UniCD/UniCD-repro
PY=/home/jgxy/anaconda3/envs/baiwei/bin/python

echo "[$(date)] === Improved Pipeline Start ==="

# Stage 1: Improved pretrain (30 epochs)
echo "[$(date)] Training improved_pretrain..."
$PY tools/train.py --config configs_improved/pretrain_pr.yaml 2>&1 | tee logs/imp_pretrain.log

# Stage 2: Improved joint (50 epochs)
echo "[$(date)] Training improved_joint..."
$PY tools/train.py --config configs_improved/joint_pr.yaml 2>&1 | tee logs/imp_joint.log

# Stage 3: Improved real finetune (30 epochs)
echo "[$(date)] Training improved_real..."
$PY tools/train.py --config configs_improved/finetune_real_pr.yaml 2>&1 | tee logs/imp_real.log

# Stage 4: Improved eval
echo "[$(date)] Evaluating improved..."
$PY tools/eval_nuc.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/joint_pr/epoch_050.pt 2>&1 | tee logs/eval_improved.log
$PY tools/eval_det.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/joint_pr/epoch_050.pt --mode syn 2>&1 | tee -a logs/eval_improved.log
$PY tools/eval_det.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/finetune_real_pr/epoch_030.pt --mode real 2>&1 | tee -a logs/eval_improved.log

echo "[$(date)] === Improved Pipeline Complete ==="

# Stage 5: Classical NUC comparison (CPU-based, quick)
echo "[$(date)] Running classical NUC baselines..."
$PY tools/classical_nuc_baseline.py --data-root /home/jgxy/work/UniCD 2>&1 | tee logs/classical_nuc.log

# Stage 6: Speed benchmark
echo "[$(date)] Running speed benchmark..."
$PY tools/benchmark_speed.py --config configs_improved/pretrain_pr.yaml --ckpt outputs_improved/pretrain_pr/epoch_030.pt 2>&1 | tee logs/speed_benchmark.log

echo "[$(date)] === ALL IMPROVED EXPERIMENTS COMPLETE ==="
