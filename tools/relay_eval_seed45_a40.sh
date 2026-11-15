#!/bin/bash
# Evaluate the seed-45 checkpoint produced on the 3060 host, on the A40 host,
# using exactly the same data root, code and metrics as seeds 42/43/44.
set -u
cd /root/work/UniCD/UniCD-repro || exit 1
P=/root/anaconda3/bin/python
CFG=configs_multiseed/seed_45_10ep.yaml
CKPT=outputs/multiseed/seed_45_from3060/epoch_010.pt
OUT=results/multiseed_10ep
LOG=logs/multiseed/relay_seed45.log

echo "[$(date '+%F %T')] RELAY_EVAL_START" >> "$LOG"
if [ ! -s "$CKPT" ]; then echo "CKPT_MISSING" >> "$LOG"; exit 1; fi
if [ -s "$OUT/seed45_det_syn.json" ]; then
  echo "SEED45_ALREADY_EVALUATED skip" >> "$LOG"; exit 0
fi

$P tools/eval_det.py --config "$CFG" --ckpt "$CKPT" --mode syn \
   --split test --score-thr 0.5 --data-root /root/work/UniCD \
   > "$OUT/seed45_det_syn.json" 2> "$LOG.syn.err"
$P tools/eval_det.py --config "$CFG" --ckpt "$CKPT" --mode real \
   --split test --score-thr 0.5 --data-root /root/work/UniCD \
   > "$OUT/seed45_det_real.json" 2> "$LOG.real.err"
$P tools/eval_nuc.py --config "$CFG" --ckpt "$CKPT" --split test \
   --data-root /root/work/UniCD \
   > "$OUT/seed45_nuc.json" 2> "$LOG.nuc.err"
$P tools/summarize_multiseed.py --dir "$OUT" --out "$OUT/summary.json" \
   >> "$LOG" 2>&1
echo "[$(date '+%F %T')] RELAY_EVAL_DONE" >> "$LOG"
