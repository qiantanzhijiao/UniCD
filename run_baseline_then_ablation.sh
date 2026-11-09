#!/usr/bin/env bash
# Baseline chain + Ablation experiments - fully automated
set -uo pipefail
cd /root/work/UniCD/UniCD-repro
PY=/root/anaconda3/bin/python3

echo "[$(date)] === Baseline + Ablation Pipeline ==="

# Phase 1: Baseline joint (already running, wait for it)
echo "[$(date)] Waiting for baseline joint to complete..."
while true; do
    if [ -f outputs/joint/epoch_050.pt ]; then
        echo "[$(date)] Baseline joint complete!"
        break
    fi
    sleep 300  # check every 5 min
done

# Phase 2: Baseline real finetune (30 epochs)
echo "[$(date)] Training baseline_real..."
$PY tools/train.py --config configs/finetune_real.yaml 2>&1 | tee logs/real_train.log

# Phase 3: Baseline eval
echo "[$(date)] Evaluating baseline..."
$PY tools/eval_nuc.py --config configs/eval.yaml --ckpt outputs/joint/epoch_050.pt 2>&1 | tee logs/eval_baseline.log
$PY tools/eval_det.py --config configs/eval.yaml --ckpt outputs/joint/epoch_050.pt --mode syn 2>&1 | tee -a logs/eval_baseline.log
$PY tools/eval_det.py --config configs/eval.yaml --ckpt outputs/real/epoch_030.pt --mode real 2>&1 | tee -a logs/eval_baseline.log

echo "[$(date)] === Baseline COMPLETE ==="

# Phase 4: Ablation 1 - No gate, no perturbation (residual only)
echo "[$(date)] Training abl_joint_nogate..."
$PY tools/train.py --config configs_improved/joint_pr_nogate.yaml 2>&1 | tee logs/abl_nogate.log
$PY tools/eval_nuc.py --config configs_improved/eval_pr_nogate.yaml --ckpt outputs_improved/joint_pr_nogate/epoch_050.pt 2>&1 | tee logs/abl_nogate_eval.log
$PY tools/eval_det.py --config configs_improved/eval_pr_nogate.yaml --ckpt outputs_improved/joint_pr_nogate/epoch_050.pt --mode syn 2>&1 | tee -a logs/abl_nogate_eval.log
cp outputs_improved/eval_nogate/eval_nuc/metrics.json results/abl_nogate__nuc_syn.json 2>/dev/null
cp outputs_improved/eval_nogate/eval_det/metrics_syn.json results/abl_nogate__det_syn.json 2>/dev/null

echo "[$(date)] === Ablation 1 COMPLETE ==="

# Phase 5: Ablation 2 - No perturbation (residual + gate)
echo "[$(date)] Training abl_joint_noperturb..."
$PY tools/train.py --config configs_improved/joint_pr_noperturb.yaml 2>&1 | tee logs/abl_noperturb.log
$PY tools/eval_nuc.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/joint_pr_noperturb/epoch_050.pt 2>&1 | tee logs/abl_noperturb_eval.log
$PY tools/eval_det.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/joint_pr_noperturb/epoch_050.pt --mode syn 2>&1 | tee -a logs/abl_noperturb_eval.log
cp outputs_improved/eval/eval_nuc/metrics.json results/abl_noperturb__nuc_syn.json 2>/dev/null
cp outputs_improved/eval/eval_det/metrics_syn.json results/abl_noperturb__det_syn.json 2>/dev/null

echo "[$(date)] === Ablation 2 COMPLETE ==="

# Phase 6: Collect all results
echo "[$(date)] Collecting results..."
$PY tools/collect_results.py --results results 2>/dev/null || echo "No collect_results.py"

echo "[$(date)] === ALL BASELINE + ABLATION EXPERIMENTS COMPLETE ==="
