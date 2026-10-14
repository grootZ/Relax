#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next 8L VL GRPO on the real lmms-lab multimodal-open-r1 dataset.
# Model / parallel / optimizer / sglang config follow the 8L smoke script
# (run-qwen38-flash-next-8l-vl-8xgpu-grpo.sh); the data + reward wiring follows
# run-qwen35-35B-A3B-8xklx.sh (openr1mm reward on the converted parquet with
# prompt / image / label columns).
#
# Usage (defaults point at /data1/zhujinlei/models):
#   MODEL_DIR=/data1/zhujinlei/models \
#   DATA_DIR=/data1/zhujinlei/models/lmms-lab/multimodal-open-r1-8k-verified/data \
#   bash scripts/training/multimodal/run-qwen38-flash-next-8l-vl-8xgpu-grpo-openr1mm.sh
#
# NOTE(agent): rollout/context lengths and batch sizes below are smoke-scale
# defaults for the reduced 8L checkpoint on one 8-GPU node; they are not tuned
# for convergence and may need adjustment after a first run.

export MODEL_DIR=/data1/zhujinlei/models
export DATA_DIR=/data1/zhujinlei/models/lmms-lab/multimodal-open-r1-8k-verified/data
export WANDB_API_KEY=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
unset http_proxy && unset https_proxy

# SGLang 的 NUMA 绑定 v2 用 `numactl --cpunodebind=N --membind=N` 硬绑定内存节点。
# Qwen4-Exp 的 PLE n-gram 表走 CPU pinned offload, 每个 TP rank 需要 11.9GiB
# cudaHostAlloc; 在硬 membind 下 node1 上的 rank(4-7) 会直接返回
# cudaErrorInvalidValue ("CUDA error: invalid argument")。v1 用软策略
# numa_set_preferred, 允许跨节点回退, 8 个 rank 都能分配成功。
export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"

# 训练侧 PLE n-gram 表走 CPU pinned（patch bundle 的开关,
# per_layer_embedding.py:292 -> PinnedVocabParallelEmbedding）。
# 默认关闭: 开启后表变成 register_buffer, PinnedVocabParallelEmbedding 这个 module
# 就一个 Parameter 都没有, 而 Megatron-Bridge 的 HF 载入走到
# param_mapping.py:1637 `next(megatron_module.parameters()).device` 会抛
# StopIteration（actor init 失败）。显存改用下面 TRAINABLE_ARGS 冻结 PLE 表来省:
# 表仍在显存（11.92GiB/rank）但不进优化器, 省掉 fp32 master 23.84GiB + 两个动量。
# 必须经 Ray runtime env 传给 worker（见 scripts/entrypoint/local.sh）,
# 交互式 export 到不了 Ray actor。
export QWEN48_PLE_CPU_OFFLOAD="${QWEN48_PLE_CPU_OFFLOAD:-0}"

set -ex
set -o pipefail

export RAY_NO_WAIT="${RAY_NO_WAIT-1}"

now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen38-flash-next-8l.sh"

MODEL_DIR="${MODEL_DIR:-/data1/zhujinlei/models}"
DATA_DIR="${DATA_DIR:-/data1/zhujinlei/models/lmms-lab/multimodal-open-r1-8k-verified/data}"
MODEL_NAME="${VL_MODEL_NAME:-Qwen3.8-Flash-Next-8L-VL}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/train-00000-of-00001-converted.parquet}"

: "${MODEL_DIR:?Set MODEL_DIR to the 8L VL model root}"
[ -d "${MODEL_DIR}/${MODEL_NAME}" ] || { echo "missing model dir ${MODEL_DIR}/${MODEL_NAME}" >&2; exit 2; }
[ -f "${PROMPT_DATA}" ] || { echo "missing prompt parquet ${PROMPT_DATA}" >&2; exit 2; }

: "${NO_SAVE:=1}"
: "${NUM_ROLLOUT:=200}"
if [[ "${NO_SAVE}" != 0 && "${NO_SAVE}" != 1 ]]; then echo "NO_SAVE must be 0 or 1" >&2; exit 2; fi
if [[ "${NO_SAVE}" == 0 ]]; then : "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"; fi

PROJECT_NAME="${PROJECT_NAME:-Relax/rl/qwen38-flash-next-vl}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-8l-vl-grpo-openr1mm-8xgpu}"

TP_SIZE="${TP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --ref-load "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
)
# if [[ "${NO_SAVE}" == 0 ]]; then
#     CKPT_ARGS+=(
#         --load "${SAVE_DIR}/${EXP_NAME}"
#         --save "${SAVE_DIR}/${EXP_NAME}"
#         --save-interval "${SAVE_INTERVAL:-20}"
#         --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
#     )
# fi

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type openr1mm
    --multimodal-keys '{"image":"image"}'
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-1024}"
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}"
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-2048}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-3072}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
    --use-streaming-dataset
    --system-prompt "${SYSTEM_PROMPT}"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-coef 0.0
    --kl-loss-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --micro-batch-size 1
    --max-tokens-per-gpu "${TOKEN_BUDGET:-4096}"
    --log-probs-max-tokens-per-gpu "${TOKEN_BUDGET:-4096}"
    --use-dynamic-batch-size
    --no-rope-fusion

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)
if [ "${CP_SIZE}" -gt 1 ]; then
    PARALLEL_ARGS+=(--calculate-per-token-loss)
fi

# Keep the QSA indexer frozen (matches the 8L smoke non-gate path).
# 同时冻结 PLE n-gram 表: 它占每 rank 9.36B 参数里的 6.40B(68%), 不冻结的话
# precision-aware optimizer 要为它分 fp32 master 23.84GiB + exp_avg/exp_avg_sq,
# optimizer step 直接 OOM（optimizer.py:817 "Tried to allocate 23.84 GiB"）。
# patch bundle 本身也是把这张表当冻结表用的。
TRAINABLE_ARGS=(
    --freeze-params-name-list
    'self_attention\.indexer\.'
    'per_layer_embedding\.ple_embedding\.ngram_embedding\.'
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)
# NOTE(agent): 本模型开 --optimizer-cpu-offload 的前提是先把 PLE n-gram 表移出优化器。
# 别被 "8L" 误导: 每 rank 9.36B 参数, 其中 PLE n-gram 表(TP8 分片 40000192x160)
# 占 6.40B = 68%。precision-aware optimizer 要为它存 fp32 master + exp_avg +
# exp_avg_sq, 合计约 105GiB/rank:
#   - GPU 侧: optimizer step 申请 PLE 的 fp32 master 23.84GiB 直接 OOM
#     (optimizer.py:817 "Tried to allocate 23.84 GiB")
#   - host 侧: HybridDeviceOptimizer 对 master weight + Adam state 做
#     param.pin_memory(), 叠加 8 个 rollout rank 各钉 11.9GiB 的 PLE 表(共 95GiB)
#     与 143GiB ckpt 的 page cache, cudaHostAlloc 返回 cudaErrorInvalidValue
#     ("CUDA error: invalid argument", hybrid_optimizer.py:280)
# 对比 run-qwen35-35B-A3B-8xklx.sh: 那个模型没有 PLE 表, 每 rank 参数约为本模型
# 的一半, 所以同样两个 flag 在那边没问题。
# 解法是上面的 QWEN48_PLE_CPU_OFFLOAD=1: PinnedVocabParallelEmbedding 用
# register_buffer 而不是 Parameter, 表既不占显存也不进优化器。

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-ep-size 8
    --sglang-mem-fraction-static 0.5
    --sglang-disable-cuda-graph
)

TRACKING_ARGS=(
    --tb-experiment-name qwen3.8-flash-next
   --use-wandb
   --wandb-project qwen3.8-flash-next
   --wandb-group qwen3.8-flash-next-${now}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
   --no-use-metrics-service
)

MISC_ARGS=(
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --selective-offload
    --use-health-check
)

mkdir -p "${SCRIPT_DIR}/log"

ray job submit --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${TRAINABLE_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
    # "${TRACKING_ARGS[@]}" \