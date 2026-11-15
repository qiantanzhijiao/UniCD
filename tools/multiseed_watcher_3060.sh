#!/bin/bash
# Deploy on the 3060 host. Waits locally for seed_44 / seed_45 checkpoints,
# then evaluates them under the SAME matched reduced-budget protocol as
# seed_42 (default_20ep) and seed_43. Running locally means it survives even
# if sshd becomes unresponsive again.
set -u
P=/home/jgxy/anaconda3/envs/baiwei/bin/python
cd /home/jgxy/work/UniCD/UniCD-repro || exit 1
DATA=/home/jgxy/work/UniCD
OUT=results/multiseed
mkdir -p "$OUT" logs/multiseed

echo "WATCHER_3060_START $(date)"

for s in 44 45; do
  CFG="configs_multiseed/seed_${s}.yaml"
  CKPT="outputs/multiseed/seed_${s}/epoch_020.pt"
  echo "waiting for seed_${s} checkpoint: $CKPT"
  while [ ! -f "$CKPT" ]; do
    sleep 120
  done
  # make sure the writer has finished before loading
  s1=$(stat -c %s "$CKPT")
  sleep 30
  s2=$(stat -c %s "$CKPT")
  while [ "$s1" != "$s2" ]; do
    s1=$s2
    sleep 30
    s2=$(stat -c %s "$CKPT")
  done

  echo "evaluating seed_${s} $(date)"
  $P tools/eval_det.py --config "$CFG" --ckpt "$CKPT" --mode syn \
     --split test --score-thr 0.5 --data-root "$DATA" \
     > "$OUT/seed_${s}_det_syn.json" 2>&1
  $P tools/eval_det.py --config "$CFG" --ckpt "$CKPT" --mode real \
     --split test --score-thr 0.5 --data-root "$DATA" \
     > "$OUT/seed_${s}_det_real.json" 2>&1
  $P tools/eval_nuc.py --config "$CFG" --ckpt "$CKPT" --split test \
     --data-root "$DATA" > "$OUT/seed_${s}_nuc.json" 2>&1
  echo "SEED_${s}_EVAL_DONE $(date)"
done

echo "=== ALL SEED RESULTS ==="
for f in "$OUT"/seed_4*.json; do
  echo "--- $f"
  cat "$f"
done
echo "WATCHER_3060_ALL_DONE $(date)"
