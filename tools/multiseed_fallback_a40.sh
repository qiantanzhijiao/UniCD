#!/bin/bash
# Autonomous A40 fallback for the multi-seed study.
#
# The 3060 host (seed_44 / seed_45, 20-epoch protocol) became unreachable at
# sshd level.  This chain completes the study entirely on the A40 using a
# MATCHED 10-EPOCH reduced-budget protocol for all four seeds (42..45):
# identical joint-stage config, identical pretrain init
# (outputs_improved/pretrain_pr/epoch_030.pt), identical 3000 iters/epoch,
# identical evaluation.  Only the random seed differs.
#
# If results/ABORT_FALLBACK exists, the chain stops at the next checkpoint
# boundary (used when the 3060 host recovers and delivers its own results).
set -u
P=/root/anaconda3/bin/python
ROOT=/root/work/UniCD/UniCD-repro
DATA=/root/work/UniCD
cd "$ROOT" || exit 1
mkdir -p logs/multiseed results/multiseed_10ep
FLAG="$ROOT/results/ABORT_FALLBACK"
CHAIN=logs/multiseed/fallback_chain.log

log() { echo "[$(date '+%F %T')] $*" >> "$CHAIN"; }
aborted() { [ -f "$FLAG" ]; }

eval_seed() {
  tag=$1; cfg=$2; ckpt=$3
  log "EVAL $tag START"
  $P tools/eval_det.py --config "$cfg" --ckpt "$ckpt" --mode syn \
     --split test --score-thr 0.5 --data-root "$DATA" \
     > "results/multiseed_10ep/${tag}_det_syn.json" \
     2> "logs/multiseed/${tag}_det_syn.err"
  $P tools/eval_det.py --config "$cfg" --ckpt "$ckpt" --mode real \
     --split test --score-thr 0.5 --data-root "$DATA" \
     > "results/multiseed_10ep/${tag}_det_real.json" \
     2> "logs/multiseed/${tag}_det_real.err"
  $P tools/eval_nuc.py --config "$cfg" --ckpt "$ckpt" --split test \
     --data-root "$DATA" \
     > "results/multiseed_10ep/${tag}_nuc.json" \
     2> "logs/multiseed/${tag}_nuc.err"
  $P tools/summarize_multiseed.py --dir results/multiseed_10ep \
     --out results/multiseed_10ep/summary.json \
     >> "$CHAIN" 2>&1
  log "EVAL $tag DONE"
}

log "CHAIN_START (10-epoch matched protocol)"

# --- seed 42: checkpoint already exists, evaluate immediately -------------
eval_seed seed42 configs_sensitivity/default_20ep.yaml \
          outputs/sensitivity/default_20ep/epoch_010.pt

# --- seed 43: wait for its epoch_010 checkpoint ---------------------------
while [ ! -f outputs/multiseed/seed_43/epoch_010.pt ]; do
  if aborted; then log "ABORT while waiting for seed43"; exit 0; fi
  sleep 60
done
# Make sure the checkpoint writer has finished.
s1=$(stat -c %s outputs/multiseed/seed_43/epoch_010.pt); sleep 30
s2=$(stat -c %s outputs/multiseed/seed_43/epoch_010.pt)
while [ "$s1" != "$s2" ]; do
  s1=$s2; sleep 30
  s2=$(stat -c %s outputs/multiseed/seed_43/epoch_010.pt)
done
log "seed43 epoch_010 ready; stopping the 20-epoch run to free RAM"
pkill -f "train.py --config configs_multiseed/seed_43.yaml" || true
pkill -f "multiseed_watcher_a40.sh" || true
sleep 20
log "seed43 20-epoch run stopped"

if aborted; then log "ABORT before seed43 eval"; exit 0; fi
eval_seed seed43 configs_multiseed/seed_43.yaml \
          outputs/multiseed/seed_43/epoch_010.pt

# --- seed 44: 10 matched epochs ------------------------------------------
if aborted; then log "ABORT before seed44"; exit 0; fi
log "TRAIN seed44 START"
$P tools/train.py --config configs_multiseed/seed_44_10ep.yaml \
   > logs/multiseed/seed_44_10ep_train.log 2>&1
log "TRAIN seed44 DONE"
if [ -f outputs/multiseed/seed_44/epoch_010.pt ]; then
  eval_seed seed44 configs_multiseed/seed_44_10ep.yaml \
            outputs/multiseed/seed_44/epoch_010.pt
else
  log "ERROR seed44 checkpoint missing"
fi

# --- seed 45: 10 matched epochs ------------------------------------------
if aborted; then log "ABORT before seed45"; exit 0; fi
log "TRAIN seed45 START"
$P tools/train.py --config configs_multiseed/seed_45_10ep.yaml \
   > logs/multiseed/seed_45_10ep_train.log 2>&1
log "TRAIN seed45 DONE"
if [ -f outputs/multiseed/seed_45/epoch_010.pt ]; then
  eval_seed seed45 configs_multiseed/seed_45_10ep.yaml \
            outputs/multiseed/seed_45/epoch_010.pt
else
  log "ERROR seed45 checkpoint missing"
fi

log "ALL_MULTISEED_DONE"
