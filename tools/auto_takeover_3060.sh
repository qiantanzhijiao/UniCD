#!/bin/bash
# Local watcher: as soon as the 3060 sshd answers again, deploy and start the
# remote multiseed watcher automatically so no second user action is needed.
ASK=/tmp/askpass_new.sh
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR"
SRC="C:/Users/Administrator/WorkBuddy/2026-11-02-21-27-15/tmp_transfer/multiseed_watcher_3060.sh"
DST="/home/jgxy/work/UniCD/UniCD-repro/tools/multiseed_watcher_3060.sh"

for i in $(seq 1 90); do
  export SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
  if timeout 30 ssh $SSHOPT jgxy@192.168.0.4 'echo SSH_PROBE_OK' 2>/dev/null | grep -q SSH_PROBE_OK; then
    echo "3060_RECOVERED $(date) attempt=$i"
    scp $SSHOPT "$SRC" "jgxy@192.168.0.4:$DST" && echo SCP_WATCHER_OK
    timeout 60 ssh $SSHOPT jgxy@192.168.0.4 \
      "cd /home/jgxy/work/UniCD/UniCD-repro && mkdir -p logs && chmod +x tools/multiseed_watcher_3060.sh && nohup bash tools/multiseed_watcher_3060.sh > logs/multiseed/watcher_3060.log 2>&1 < /dev/null & echo WATCHER_STARTED; echo '===STATUS==='; uptime; free -h; echo '===CKPTS==='; ls -la outputs/multiseed/seed_44/ outputs/multiseed/seed_45/ 2>/dev/null | tail -20; echo '===TRAIN PROCS==='; ps aux | grep -E 'train.py|eval_gate_scan' | grep -v grep | head" 2>&1
    echo "TAKEOVER_DONE $(date)"
    exit 0
  fi
  echo "attempt $i not ready $(date +%H:%M:%S)"
  sleep 20
done
echo "TAKEOVER_TIMEOUT: 3060 never came back"
exit 1
