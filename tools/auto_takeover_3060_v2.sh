#!/bin/bash
# Poll the 3060 host until sshd answers.  On recovery: kill the stuck
# pre-loading gate-scan process, deploy the on-host multi-seed watcher, and
# pull whatever results already exist.  The A40 fallback chain is NOT aborted:
# it delivers the matched 10-epoch protocol; 3060 results (20-epoch) are kept
# as a supplementary robustness check.
cd "C:/Users/Administrator/WorkBuddy/2026-11-02-21-27-15" || exit 1
mkdir -p tmp_transfer/results_3060
OLD=/tmp/askpass_old.sh
NEW=/tmp/askpass_new.sh
LOG=/tmp/auto_takeover_3060_v2.log
: > "$LOG"

echo "START $(date '+%F %T')" >> "$LOG"
for i in $(seq 1 600); do
  export SSH_ASKPASS="$NEW" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
  OUT=$(timeout 20 ssh -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 \
        -o BatchMode=no -o LogLevel=ERROR jgxy@192.168.0.4 \
        'echo SSH_OK; date "+%F %T"; uptime' 2>&1)
  if echo "$OUT" | grep -q SSH_OK; then
    echo "RECOVERED attempt=$i $(date '+%F %T')" >> "$LOG"
    echo "$OUT" >> "$LOG"

    # 1. kill the old pre-loading gate scan (the cause of the sshd stall)
    timeout 60 ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR jgxy@192.168.0.4 \
      "pkill -9 -f '[e]val_gate_scan.py' 2>/dev/null; echo GATE_SCAN_KILLED; \
       free -h; ps -eo pid,stat,etime,cmd | grep '[t]rain.py' | head -10" >> "$LOG" 2>&1

    # 2. deploy and launch the on-host watcher (survives future sshd stalls)
    scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR tmp_transfer/multiseed_watcher_3060.sh \
      jgxy@192.168.0.4:/home/jgxy/work/UniCD/UniCD-repro/tools/multiseed_watcher_3060.sh >> "$LOG" 2>&1
    timeout 60 ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR jgxy@192.168.0.4 \
      "cd /home/jgxy/work/UniCD/UniCD-repro; chmod +x tools/multiseed_watcher_3060.sh; \
       nohup bash tools/multiseed_watcher_3060.sh > logs/multiseed/watcher_3060.log 2>&1 < /dev/null & \
       echo WATCHER_3060_LAUNCHED; \
       echo '=== seed44 ==='; tail -2 logs/multiseed/seed_44.log 2>/dev/null; \
       echo '=== seed45 ==='; tail -2 logs/multiseed/seed_45.log 2>/dev/null; \
       echo '=== ckpts ==='; ls outputs/multiseed/*/epoch_020.pt 2>/dev/null" >> "$LOG" 2>&1

    # 3. pull any already-finished results
    scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR "jgxy@192.168.0.4:/home/jgxy/work/UniCD/UniCD-repro/results/multiseed/*.json" \
      tmp_transfer/results_3060/ >> "$LOG" 2>&1
    echo "3060_TAKEOVER_DONE $(date '+%F %T')" >> "$LOG"
    exit 0
  fi
  echo "attempt $i not ready $(date '+%H:%M:%S')" >> "$LOG"
  sleep 40
done
echo "TAKEOVER_TIMEOUT_V2 $(date '+%F %T')" >> "$LOG"
exit 1
