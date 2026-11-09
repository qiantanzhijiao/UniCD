#!/bin/bash
cd /root/work/UniCD/UniCD-repro || exit 1
export PY=/home/jgxy/anaconda3/envs/baiwei/bin/python
export RUN_BASELINE=1
export RUN_IMPROVED=1
export RUN_ABLATIONS=0
mkdir -p logs
nohup ./run_unied_pipeline.sh > logs/pipeline_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PIPELINE_PID=$!"
