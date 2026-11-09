#!/bin/bash
cd /root/work/UniCD/UniCD-repro || exit 1
mkdir -p logs outputs
pkill -f 'prepare_coefs.py' 2>/dev/null
sleep 1
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/prepare_coefs.py \
  --syn-root /root/work/UniCD/IRBFD-syn/VOC2007 \
  --out outputs/coef_cache_syn.npz --split trainval \
  > logs/prepare_coefs_run.log 2>&1
echo "EXIT=$?"
