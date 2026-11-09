#!/usr/bin/env bash
# 完整实验脚本：主实验 + 消融实验 + 结果收集
# 用法: nohup bash run_full_experiment.sh >> logs/full_experiment.log 2>&1 &

set -euo pipefail
cd /root/work/UniCD/UniCD-repro
PY=/home/jgxy/anaconda3/envs/baiwei/bin/python

echo "[$(date)] === Phase 1: Main experiments (baseline + improved) ==="
RUN_BASELINE=1 RUN_IMPROVED=1 RUN_ABLATIONS=0 PY=$PY bash run_unied_pipeline.sh

echo "[$(date)] === Phase 2: Ablation experiments ==="
# Ablation 1: + residual, no gate, no perturbation
echo "[$(date)] Training abl_pretrain_nogate..."
$PY tools/train.py --config configs_improved/pretrain_pr_nogate.yaml
echo "[$(date)] Training abl_joint_nogate..."
$PY tools/train.py --config configs_improved/joint_pr_nogate.yaml
echo "[$(date)] Evaluating abl_nogate..."
$PY tools/eval_nuc.py --config configs_improved/eval_pr_nogate.yaml --ckpt outputs_improved/joint_pr_nogate/epoch_050.pt
$PY tools/eval_det.py --config configs_improved/eval_pr_nogate.yaml --ckpt outputs_improved/joint_pr_nogate/epoch_050.pt --mode syn
cp outputs_improved/eval_nogate/eval_nuc/metrics.json results/abl_nogate__nuc_syn.json 2>/dev/null || true
cp outputs_improved/eval_nogate/eval_det/metrics_syn.json results/abl_nogate__det_syn.json 2>/dev/null || true

# Ablation 2: + residual + gate, no perturbation  
echo "[$(date)] Training abl_joint_noperturb..."
$PY tools/train.py --config configs_improved/joint_pr_noperturb.yaml
echo "[$(date)] Evaluating abl_noperturb..."
$PY tools/eval_nuc.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/joint_pr_noperturb/epoch_050.pt
$PY tools/eval_det.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/joint_pr_noperturb/epoch_050.pt --mode syn
cp outputs_improved/eval/eval_nuc/metrics.json results/abl_noperturb__nuc_syn.json 2>/dev/null || true
cp outputs_improved/eval/eval_det/metrics_syn.json results/abl_noperturb__det_syn.json 2>/dev/null || true

echo "[$(date)] === Phase 3: Collect results ==="
$PY tools/collect_results.py --results results

echo "[$(date)] === ALL EXPERIMENTS COMPLETE ==="
