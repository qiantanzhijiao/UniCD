#!/bin/bash
cd /root/work/UniCD/UniCD-repro
echo "[$(date)] Starting joint training..."
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/train.py --config configs/joint_train.yaml >> logs/joint_train.log 2>&1
echo "[$(date)] Joint training done."
echo "[$(date)] Starting real finetune..."
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/train.py --config configs/finetune_real.yaml >> logs/real_train.log 2>&1
echo "[$(date)] Real finetune done."
echo "[$(date)] Evaluating baseline..."
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/eval_nuc.py --config configs/eval.yaml --ckpt outputs/joint/epoch_050.pt >> logs/eval_baseline.log 2>&1
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/eval_det.py --config configs/eval.yaml --ckpt outputs/joint/epoch_050.pt --mode syn >> logs/eval_baseline.log 2>&1
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/eval_det.py --config configs/eval.yaml --ckpt outputs/real/epoch_030.pt --mode real >> logs/eval_baseline.log 2>&1
echo "[$(date)] Starting improved pretrain..."
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/train.py --config configs_improved/pretrain_pr.yaml >> logs/imp_pretrain.log 2>&1
echo "[$(date)] Starting improved joint..."
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/train.py --config configs_improved/joint_pr.yaml >> logs/imp_joint.log 2>&1
echo "[$(date)] Starting improved real finetune..."
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/train.py --config configs_improved/finetune_real_pr.yaml >> logs/imp_real.log 2>&1
echo "[$(date)] Evaluating improved..."
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/eval_nuc.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/joint_pr/epoch_050.pt >> logs/eval_improved.log 2>&1
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/eval_det.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/joint_pr/epoch_050.pt --mode syn >> logs/eval_improved.log 2>&1
/home/jgxy/anaconda3/envs/baiwei/bin/python tools/eval_det.py --config configs_improved/eval_pr.yaml --ckpt outputs_improved/finetune_real_pr/epoch_030.pt --mode real >> logs/eval_improved.log 2>&1
echo "[$(date)] ALL DONE."
