#!/usr/bin/env bash
# 七组单变量消融：除算法开关外超参完全一致。
# 训练集 = MATH level 4-5（竞赛级）；评测 = olympiad(AIME+AMC+MATH-L5, 分布外) + math_l45(分布内)
# 用法: bash scripts/run_ablations.sh
# 环境变量可覆盖: BASE_MODEL / TRAIN_SET / EVAL_SET / MAX_STEPS / GROUP_SIZE / GROUP_PER_STEP
#                 TRAIN_SAMPLES / EVAL_SAMPLES / MAX_NEW_TOKENS / DEVICE
set -e

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TRAIN_SET=${TRAIN_SET:-math_l45}
EVAL_SET=${EVAL_SET:-olympiad,math_l45}
MAX_STEPS=${MAX_STEPS:-150}
GROUP_SIZE=${GROUP_SIZE:-8}
GROUP_PER_STEP=${GROUP_PER_STEP:-1}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3000}
EVAL_SAMPLES=${EVAL_SAMPLES:-150}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-384}
DEVICE=${DEVICE:-auto}

for algo in grpo clip_higher dynamic token_level overlong dapo entropy_reg; do
    echo "=== training: ${algo} ==="
    python scripts/train.py --algorithm "${algo}" \
        --base-model "${BASE_MODEL}" \
        --dataset "${TRAIN_SET}" \
        --eval-dataset "${EVAL_SET}" \
        --train-samples "${TRAIN_SAMPLES}" \
        --eval-samples "${EVAL_SAMPLES}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --device "${DEVICE}" \
        --max-steps "${MAX_STEPS}" \
        --group-size "${GROUP_SIZE}" \
        --groups-per-step "${GROUP_PER_STEP}" \
        --run-name "${algo}"
done

echo "All ablations complete."
