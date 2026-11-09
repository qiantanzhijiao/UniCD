#!/usr/bin/env bash
# ============================================================================
# UniCD-PR 全流程自动化脚本 (Ubuntu 18.04 / /root/work/UniCD)
#
# 功能:
#   1) 确保 GT 系数缓存就绪(若你正在跑 prepare_coefs.py, 会自动等待其结束)
#   2) 按序自动运行: 基线三步训练 -> 改进版三步训练 -> (可选)消融 -> 全部评估
#   3) 汇总论文数据(PSNR/SSIM/SCRG/P/R/AP50 + 训练耗时), 输出到 results/ 下
#      summary.md / summary.csv / timing.tsv / 各指标 json / gpu 与 env 信息
#
# 用法:
#   ./run_unied_pipeline.sh            # 默认: 基线 + 改进 + 评估 + 汇总
#   ./run_unied_pipeline.sh check      # 只检查环境/数据/缓存/配置, 不执行
#
# 环境变量(均可覆盖默认):
#   RUN_BASELINE=0/1   是否跑论文基线(configs/ 三步)          默认 1
#   RUN_IMPROVED=0/1   是否跑改进方案(configs_improved/ 三步)  默认 1
#   RUN_ABLATIONS=0/1  是否跑消融(联合训练阶段)               默认 0
#   SMOKE=1            用冒烟配置快速验证全流程(数字无意义)
#   FORCE_RERUN=1      忽略"已存在则跳过"检查, 全部重跑
#   PY=python          解释器(默认 python, 即 Anaconda 3.6)
#
# 预期耗时(RTX3060, 640x512, batch4): 基线+改进约 2-5 天; 消融另加约 1-2 天。
# 中断后重跑脚本会自动跳过已完成阶段(以输出目录内的 epoch_*.pt 为准)。
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. 路径与开关
# ---------------------------------------------------------------------------
ROOT=${ROOT:-/root/work/UniCD}
REPRO="$ROOT/UniCD-repro"
PY=${PY:-python}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_IMPROVED=${RUN_IMPROVED:-1}
RUN_ABLATIONS=${RUN_ABLATIONS:-0}
SMOKE=${SMOKE:-0}
FORCE_RERUN=${FORCE_RERUN:-0}

COEF_CACHE="$REPRO/outputs/coef_cache_syn.npz"
SYN_ROOT="$ROOT/IRBFD-syn/VOC2007"
REAL_ROOT="$ROOT/IRBFD-real/VOC2007"

LOGS="$REPRO/logs"
RESULTS="$REPRO/results"
mkdir -p "$LOGS" "$RESULTS"

LOG="$LOGS/pipeline_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG"

# 所有命令的输出同时进终端与日志; 失败立即终止整个脚本
run() {
    echo "### $*"
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

now() { date "+%Y-%m-%d %H:%M:%S"; }
info() { echo "[$(now)] $*"; }
die()  { echo "[$(now)] FATAL: $*" >&2; exit 1; }

# 所有相对路径(配置里的 output.save_dir 等)都以 UniCD-repro 为基准
cd "$REPRO" || die "无法进入目录: $REPRO"

# 阶段计时 -> results/timing.tsv
_t0=0
tick() { _t0=$(date +%s); }
tock() {
    local tag=$1 dt=$(( $(date +%s) - _t0 ))
    printf "%s\t%d\n" "$tag" "$dt" >> "$RESULTS/timing.tsv"
    info "stage[$tag] done in ${dt}s"
}

# 阶段完成标志: 输出目录内是否已有最终 epoch 权重
done_ckpt() { [ -f "$REPRO/$1/epoch_$(printf '%03d' "$2").pt" ]; }

require_dir()  { [ -d "$1" ] || die "目录不存在: $1 (请确认数据已拷贝到 /root/work/UniCD)"; }
require_file() { [ -f "$1" ] || die "缺少文件: $1"; }

# ---------------------------------------------------------------------------
# 1. 环境检查
# ---------------------------------------------------------------------------
check_env() {
    info "== 环境检查 =="
    require_dir "$REPRO"
    require_dir "$SYN_ROOT"
    require_dir "$REAL_ROOT"
    "$PY" -c "import torch, torchvision, numpy, scipy, PIL, yaml; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
        || die "python 环境缺少依赖, 请先按 README 3.0 离线安装"
    if nvidia-smi -L >/dev/null 2>&1; then
        nvidia-smi -L | head -1 > "$RESULTS/gpu_info.txt"
        cat "$RESULTS/gpu_info.txt"
    else
        echo "WARNING: 未检测到 NVIDIA GPU (nvidia-smi 不可用); CPU 全量训练不现实, 建议用 SMOKE=1 验证管线"
    fi
    "$PY" -c "import sys; print('python', sys.version.split()[0])" > "$RESULTS/env_info.txt"
    "$PY" -c "import torch; print('torch', torch.__version__)" >> "$RESULTS/env_info.txt"
}

# 确保系数缓存就绪: 已存在则校验; 有进程在跑则等待; 否则重新生成
# 传参 1 = check 模式: 只报告状态, 不等待也不生成
ensure_coef_cache() {
    local check_only=${1:-0}
    info "== 系数缓存 =="
    if [ -f "$COEF_CACHE" ]; then
        if "$PY" -c "
import numpy as np
d = np.load('$COEF_CACHE', allow_pickle=True)
print('coef cache ok: stems=%d coefs=%s' % (len(d['stems']), d['coefs'].shape))" >/dev/null 2>&1; then
            info "系数缓存已存在且可读, 跳过生成: $COEF_CACHE"
            return
        fi
        if [ "$check_only" = "1" ]; then
            info "WARNING: 系数缓存存在但读取失败, 正式运行时会重新生成"
            return
        fi
        info "系数缓存存在但读取失败, 将重新生成"
    fi
    if [ "$check_only" = "1" ]; then
        info "系数缓存尚未生成(正式运行时会先自动生成, 或等待正在运行的 prepare_coefs.py 结束)"
        return
    fi
    while pgrep -f "[p]repare_coefs.py" >/dev/null 2>&1; do
        info "检测到 prepare_coefs.py 正在运行, 等待其完成..."
        sleep 60
    done
    if [ -f "$COEF_CACHE" ]; then
        info "prepare_coefs 已完成, 缓存就绪"
        return
    fi
    info "未发现系数缓存, 开始生成(约 15 分钟, 一次性)"
    tick
    run "$PY" "$REPRO/tools/prepare_coefs.py" \
        --syn-root "$SYN_ROOT" \
        --out "$COEF_CACHE" \
        --split trainval
    tock "prepare_coefs"
    require_file "$COEF_CACHE"
}

# ---------------------------------------------------------------------------
# 2. 单阶段训练(带"已存在则跳过"与计时)
#    train_stage <tag> <cfg> <out_dir> <final_epoch> [resume_ckpt]
# ---------------------------------------------------------------------------
train_stage() {
    local tag=$1 cfg=$2 out_dir=$3 final_epoch=$4 resume=${5:-}
    if [ "$FORCE_RERUN" = "0" ] && done_ckpt "$out_dir" "$final_epoch"; then
        info "SKIP $tag (输出已存在: $out_dir/epoch_$(printf '%03d' "$final_epoch").pt)"
        return
    fi
    require_file "$REPRO/$cfg"
    [ -z "$resume" ] || require_file "$REPRO/$resume"
    info "== 训练: $tag ($cfg) =="
    tick
    run "$PY" "$REPRO/tools/train.py" --config "$REPRO/$cfg"
    tock "$tag"
    require_file "$(final_pt_path "$out_dir" "$final_epoch")"
}

final_pt_path() { echo "$REPRO/$1/epoch_$(printf '%03d' "$2").pt"; }

# ---------------------------------------------------------------------------
# 3. 评估(指标 json 复制到 results/{method}__{kind}.json)
#    run_eval <method> <eval_cfg> <ckpt> [real_ckpt]
#    评估工具把指标写到 {eval_cfg 的 output.save_dir}/eval_nuc 与 /eval_det
# ---------------------------------------------------------------------------
eval_out_dir() {
    "$PY" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['output']['save_dir'])" "$1" \
        || die "解析 eval 配置失败: $1"
}

run_eval() {
    local method=$1 eval_cfg=$2 ckpt=$3 real_ckpt=${4:-}
    local eout; eout=$(eval_out_dir "$REPRO/$eval_cfg")
    require_file "$REPRO/$eval_cfg" "$ckpt"
    info "== 评估: $method (cfg=$eval_cfg, ckpt=$ckpt) =="
    run "$PY" "$REPRO/tools/eval_nuc.py" --config "$REPRO/$eval_cfg" --ckpt "$ckpt"
    cp "$REPRO/$eout/eval_nuc/metrics.json" "$RESULTS/${method}__nuc_syn.json"
    run "$PY" "$REPRO/tools/eval_det.py" --config "$REPRO/$eval_cfg" --ckpt "$ckpt" --mode syn
    cp "$REPRO/$eout/eval_det/metrics_syn.json" "$RESULTS/${method}__det_syn.json"
    if [ -n "$real_ckpt" ]; then
        require_file "$real_ckpt"
        run "$PY" "$REPRO/tools/eval_det.py" --config "$REPRO/$eval_cfg" --ckpt "$real_ckpt" --mode real
        cp "$REPRO/$eout/eval_det/metrics_real.json" "$RESULTS/${method}__det_real.json"
    fi
}

# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
main() {
    check_env
    ensure_coef_cache

    # ---- 基线(poly-only, configs/) ----
    if [ "$RUN_BASELINE" = "1" ]; then
        if [ "$SMOKE" = "1" ]; then
            info "SMOKE: 基线冒烟"
            train_stage baseline_smoke configs/smoke_cpu.yaml outputs/smoke 1
            run_eval baseline_smoke configs/eval.yaml "$REPRO/outputs/smoke/epoch_001.pt"
        else
            train_stage baseline_pretrain configs/pretrain_nuc.yaml  outputs/pretrain 30
            train_stage baseline_joint    configs/joint_train.yaml   outputs/joint 50 outputs/pretrain/epoch_030.pt
            train_stage baseline_real     configs/finetune_real.yaml outputs/real 30 outputs/joint/epoch_050.pt
            run_eval baseline_poly configs/eval.yaml \
                "$REPRO/outputs/joint/epoch_050.pt" "$REPRO/outputs/real/epoch_030.pt"
        fi
    fi

    # ---- 改进版(residual+gate+perturb, configs_improved/) ----
    if [ "$RUN_IMPROVED" = "1" ]; then
        if [ "$SMOKE" = "1" ]; then
            info "SMOKE: 改进冒烟"
            train_stage improved_smoke configs_improved/smoke_pr.yaml outputs_improved/smoke_pr 1
            run_eval improved_smoke configs_improved/eval_pr.yaml \
                "$REPRO/outputs_improved/smoke_pr/epoch_001.pt"
        else
            train_stage improved_pretrain configs_improved/pretrain_pr.yaml      outputs_improved/pretrain_pr 30
            train_stage improved_joint    configs_improved/joint_pr.yaml         outputs_improved/joint_pr 50 outputs_improved/pretrain_pr/epoch_030.pt
            train_stage improved_real     configs_improved/finetune_real_pr.yaml outputs_improved/real_pr 30 outputs_improved/joint_pr/epoch_050.pt
            run_eval improved_pr configs_improved/eval_pr.yaml \
                "$REPRO/outputs_improved/joint_pr/epoch_050.pt" "$REPRO/outputs_improved/real_pr/epoch_030.pt"
        fi
    fi

    # ---- 消融(可选, 只到联合训练 + syn 评估) ----
    if [ "$RUN_ABLATIONS" = "1" ]; then
        if [ "$SMOKE" = "1" ]; then
            info "SKIP ablations (SMOKE 模式不跑消融)"
        else
            # +残差(无门控, 无扰动): 需与其一致的 gate=false 预训练
            train_stage abl_pretrain_nogate configs_improved/pretrain_pr_nogate.yaml outputs_improved/pretrain_pr_nogate 30
            train_stage abl_joint_nogate    configs_improved/joint_pr_nogate.yaml    outputs_improved/joint_pr_nogate 50 outputs_improved/pretrain_pr_nogate/epoch_030.pt
            run_eval abl_nogate configs_improved/eval_pr_nogate.yaml \
                "$REPRO/outputs_improved/joint_pr_nogate/epoch_050.pt"
            # +残差+门控(无扰动): 复用 pretrain_pr
            train_stage abl_joint_noperturb configs_improved/joint_pr_noperturb.yaml outputs_improved/joint_pr_noperturb 50 outputs_improved/pretrain_pr/epoch_030.pt
            run_eval abl_noperturb configs_improved/eval_pr.yaml \
                "$REPRO/outputs_improved/joint_pr_noperturb/epoch_050.pt"
        fi
    fi

    # ---- 汇总论文数据 ----
    info "== 汇总结果 =="
    run "$PY" "$REPRO/tools/collect_results.py" --results "$RESULTS"
    info "全部完成。日志: $LOG ; 结果目录: $RESULTS"
}

MODE=${1:-run}
case "$MODE" in
    check)
        check_env
        ensure_coef_cache 1
        info "== 配置/阶段就绪检查 =="
        for f in \
            configs/pretrain_nuc.yaml configs/joint_train.yaml configs/finetune_real.yaml \
            configs_improved/pretrain_pr.yaml configs_improved/joint_pr.yaml configs_improved/finetune_real_pr.yaml \
            configs_improved/eval_pr.yaml configs_improved/pretrain_pr_nogate.yaml \
            configs_improved/joint_pr_nogate.yaml configs_improved/joint_pr_noperturb.yaml \
            configs_improved/eval_pr_nogate.yaml \
            tools/train.py tools/eval_nuc.py tools/eval_det.py tools/collect_results.py; do
            require_file "$REPRO/$f"
        done
        info "check 通过: 数据/依赖/缓存/配置均就绪。可直接 ./run_unied_pipeline.sh"
        ;;
    run)
        main
        ;;
    *)
        die "未知参数: $MODE (支持: run / check)"
        ;;
esac
