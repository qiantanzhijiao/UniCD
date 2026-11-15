#!/bin/bash
# Poll the A40 chain for the real multi-seed summary.  As soon as four seeds
# have been evaluated, generate the LaTeX subsection, patch the paper and
# recompile.  If the fourth seed is still missing after ~8 h, fall back to
# the three completed seeds and keep the Limitations text consistent.
cd "C:/Users/Administrator/WorkBuddy/2026-11-02-21-27-15" || exit 1
LOG=/tmp/auto_fill_multiseed.log
SUM=tmp_transfer/multiseed_summary.json
: > "$LOG"
echo "START $(date '+%F %T')" >> "$LOG"

FILLED3=0
for i in $(seq 1 180); do
  export SSH_ASKPASS=/tmp/askpass_old.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
  timeout 60 ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o ConnectTimeout=20 -o LogLevel=ERROR root@10.100.9.69 \
     'cat /root/work/UniCD/UniCD-repro/results/multiseed_10ep/summary.json 2>/dev/null' \
     > "$SUM.tmp" 2>/dev/null
  N=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_seeds'])" "$SUM.tmp" 2>/dev/null)
  if [ -z "$N" ]; then N=0; fi
  echo "attempt $i n_seeds=$N $(date '+%F %T')" >> "$LOG"

  if [ "$N" -ge 4 ]; then
    cp "$SUM.tmp" "$SUM"
    python tmp_transfer/gen_multiseed_tex.py --summary "$SUM" --min-seeds 4 >> "$LOG" 2>&1
    R=$?
    echo "GEN_EXIT=$R attempt=$i" >> "$LOG"
    if [ "$R" -eq 0 ]; then echo "AUTO_FILL_DONE_4SEEDS $(date '+%F %T')" >> "$LOG"; exit 0; fi
  fi

  if [ "$N" -ge 3 ] && [ "$i" -ge 96 ] && [ "$FILLED3" -eq 0 ]; then
    cp "$SUM.tmp" "$SUM"
    python tmp_transfer/gen_multiseed_tex.py --summary "$SUM" --min-seeds 3 >> "$LOG" 2>&1
    R=$?
    echo "GEN3_EXIT=$R attempt=$i" >> "$LOG"
    if [ "$R" -eq 0 ]; then
      echo "AUTO_FILL_PROVISIONAL_3SEEDS $(date '+%F %T')" >> "$LOG"
      FILLED3=1
    fi
  fi
  sleep 300
done
echo "AUTO_FILL_TIMEOUT $(date '+%F %T')" >> "$LOG"
exit 1
