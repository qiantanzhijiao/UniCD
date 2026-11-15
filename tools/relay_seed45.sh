#!/bin/bash
# Local orchestrator: relay the 3060 seed-45 (10-epoch) checkpoint back to
# the A40 through this Windows host (the two servers are on different subnets
# and cannot reach each other), then finish the 4-seed study on the A40 with
# identical data, code and evaluation scripts.
set -u
ROOT="C:/Users/Administrator/WorkBuddy/2026-11-02-21-27-15"
cd "$ROOT" || exit 1
LOG=/tmp/relay_seed45.log
: > "$LOG"
echo "RELAY_START $(date '+%F %T')" >> "$LOG"

A40=root@10.100.9.69
S30=jgxy@192.168.0.4
RROOT=/root/work/UniCD/UniCD-repro
SROOT=/home/jgxy/work/UniCD/UniCD-repro
SCKPT=$SROOT/outputs/multiseed/seed_45_10ep/epoch_010.pt
LCKPT=$ROOT/tmp_transfer/seed45_3060_epoch010.pt
RCKPT=$RROOT/outputs/multiseed/seed_45_from3060/epoch_010.pt
OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25"
export DISPLAY=:0 SSH_ASKPASS_REQUIRE=force

sshA() { SSH_ASKPASS=/tmp/askpass_old.sh ssh $OPTS "$A40" "$@"; }
sshS() { SSH_ASKPASS=/tmp/askpass_new.sh ssh $OPTS "$S30" "$@"; }
scpS() { SSH_ASKPASS=/tmp/askpass_new.sh scp $OPTS "$@"; }
scpA() { SSH_ASKPASS=/tmp/askpass_old.sh scp $OPTS "$@"; }

# ---- 1. wait for the 3060 checkpoint (up to 9 h) --------------------------
FOUND=0
for i in $(seq 1 108); do
  if sshS "test -s '$SCKPT'" 2>/dev/null; then FOUND=1; break; fi
  echo "wait_ckpt $i $(date '+%F %T')" >> "$LOG"
  sleep 300
done
if [ "$FOUND" -ne 1 ]; then
  echo "RELAY_TIMEOUT_CKPT $(date '+%F %T')" >> "$LOG"; exit 1
fi
s1=$(sshS "stat -c %s '$SCKPT'" 2>/dev/null)
sleep 40
s2=$(sshS "stat -c %s '$SCKPT'" 2>/dev/null)
if [ "$s1" != "$s2" ]; then sleep 90; fi
echo "CKPT_READY size=$s2 $(date '+%F %T')" >> "$LOG"

# ---- 2. transfer 3060 -> Windows -> A40 -----------------------------------
scpS "$S30:$SCKPT" "$LCKPT" >> "$LOG" 2>&1 || {
  echo "SCP_FROM_3060_FAILED $(date '+%F %T')" >> "$LOG"; exit 1; }
sshA "mkdir -p $RROOT/outputs/multiseed/seed_45_from3060" >> "$LOG" 2>&1
scpA "$ROOT/tmp_transfer/relay_eval_seed45_a40.sh" \
     "$A40:$RROOT/tools/relay_eval_seed45_a40.sh" >> "$LOG" 2>&1
scpA "$LCKPT" "$A40:$RCKPT" >> "$LOG" 2>&1 || {
  echo "SCP_TO_A40_FAILED $(date '+%F %T')" >> "$LOG"; exit 1; }
echo "CKPT_RELAYED local_size=$(stat -c %s "$LCKPT") $(date '+%F %T')" >> "$LOG"

# ---- 3. wait until seed44 has been evaluated on the A40 (up to 8 h) -------
for i in $(seq 1 96); do
  if sshA "test -s $RROOT/results/multiseed_10ep/seed44_nuc.json" 2>/dev/null; then
    break
  fi
  echo "wait_seed44 $i $(date '+%F %T')" >> "$LOG"
  sleep 300
done

# ---- 4. if the A40 chain already finished seed45, keep its result ---------
HAS45=$(sshA "test -s $RROOT/results/multiseed_10ep/seed45_det_syn.json && echo YES || echo NO" 2>/dev/null)
if [ "$HAS45" = "YES" ]; then
  echo "A40_ALREADY_HAS_SEED45 keep_A40_result $(date '+%F %T')" >> "$LOG"
  exit 0
fi

# ---- 5. stop the redundant A40 seed45 run, wait for the chain to exit -----
sshA "cd $RROOT && touch results/ABORT_FALLBACK && \
      pkill -9 -f '[t]rain.py --config configs_multiseed/seed_45_10ep.yaml' 2>/dev/null; \
      echo CHAIN_STOP_REQUESTED" >> "$LOG" 2>&1
for i in $(seq 1 30); do
  ALIVE=$(sshA "pgrep -f '[m]ultiseed_fallback_a40.sh' | head -1" 2>/dev/null)
  if [ -z "$ALIVE" ]; then break; fi
  sleep 20
done
echo "CHAIN_STOPPED $(date '+%F %T')" >> "$LOG"

# ---- 6. evaluate the relayed checkpoint on the A40 ------------------------
sshA "cd $RROOT && bash tools/relay_eval_seed45_a40.sh" >> "$LOG" 2>&1
N4=$(sshA "/root/anaconda3/bin/python -c \"import json; print(json.load(open('$RROOT/results/multiseed_10ep/summary.json'))['n_seeds'])\"" 2>/dev/null)
if [ "$N4" = "4" ]; then
  touch "$ROOT/tmp_transfer/seed45_from_3060.flag"
  echo "RELAY_SUCCESS_4SEEDS $(date '+%F %T')" >> "$LOG"
else
  echo "RELAY_EVAL_INCOMPLETE n_seeds=$N4 $(date '+%F %T')" >> "$LOG"
fi
echo "RELAY_FINISHED $(date '+%F %T')" >> "$LOG"
sshA "cat $RROOT/results/multiseed_10ep/summary.json" >> "$LOG" 2>&1
exit 0
