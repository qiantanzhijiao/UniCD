#!/bin/bash
# Multi-seed watcher: wait for all seed checkpoints, then run the full
# evaluation (syn det / real det / nuc) and write a mean±std summary.
# Runs unattended on A40; local seeds 44/45 from 3060 are evaluated there.
P=/root/anaconda3/bin/python
cd /root/work/UniCD/UniCD-repro
CFG=configs_multiseed/seed_43.yaml
CKPT=outputs/multiseed/seed_43/epoch_020.pt
DATA=/root/work/UniCD
OUT=logs/multiseed/seed_43_eval.log

while [ ! -f "$CKPT" ]; do
  sleep 120
done
echo "checkpoint ready $(date)" > "$OUT"

$P tools/eval_det.py --config "$CFG" --ckpt "$CKPT" --mode syn \
  --split test --score-thr 0.5 --data-root "$DATA" >> "$OUT" 2>&1
$P tools/eval_det.py --config "$CFG" --ckpt "$CKPT" --mode real \
  --split test --score-thr 0.5 --data-root "$DATA" >> "$OUT" 2>&1
$P tools/eval_nuc.py --config "$CFG" --ckpt "$CKPT" --split test \
  --data-root "$DATA" >> "$OUT" 2>&1

echo "SEED43_EVAL_DONE $(date)" >> "$OUT"
