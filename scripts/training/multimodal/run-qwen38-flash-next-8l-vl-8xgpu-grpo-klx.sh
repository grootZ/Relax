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

set -ex
set -o pipefail
export WORKDIR="${WORKDIR:-/data1/zhujinlei/qwen38}"
export MODEL_DIR=/data1/zhujinlei
export DATA_DIR=/data1/zhujinlei/lmms-lab/multimodal-open-r1-8k-verified/data
export WANDB_API_KEY=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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RL_MODEL_TYPE="Qwen38_flash_multimodal"
export XMLIR_USE_HYDRA_LINEAR=${XMLIR_USE_HYDRA_LINEAR:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}
export XMLIR_MEMCPY_RETRY_SYNC=${XMLIR_MEMCPY_RETRY_SYNC:-true}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7}
unset http_proxy && unset https_proxy

now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
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
    --moe-grouped-gemm
    --micro-batch-size 1
    --max-tokens-per-gpu "${TOKEN_BUDGET:-4096}"
    --log-probs-max-tokens-per-gpu "${TOKEN_BUDGET:-4096}"
    --use-dynamic-batch-size
    --no-rope-fusion

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --moe-flex-dispatcher-backend deepep
    --moe-token-dispatcher-type flex
    --moe-grouped-gemm true
)
if [ "${CP_SIZE}" -gt 1 ]; then
    PARALLEL_ARGS+=(--calculate-per-token-loss)
fi

TRAINABLE_ARGS=(
    --freeze-params-name-list
    'self_attention\.indexer\.'
    'per_layer_embedding\.ple_embedding\.ngram_embedding\.'
)
# TRAINABLE_ARGS=()
# if [[ "${STAGE2_RL_GATE}" == 0 ]]; then
#     TRAINABLE_ARGS+=(--freeze-params-name-list 'self_attention\.indexer\.')
# fi

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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
    --sglang-page-size 64
    --sglang-trust-remote-code
    --sglang-attention-backend kunlun
    --sglang-disable-custom-all-reduce
    --sglang-disable-shared-experts-fusion
    --sglang-tp-size 8
    --sglang-ep-size 8
    --sglang-dtype bfloat16
    --sglang-mem-fraction-static 0.5
    # Below mirrors klx/server/qwen3.8/qwen3.8_flash_next/run_server.sh, which is
    # the known-good serving config for this model on P800 x8.
    --sglang-chunked-prefill-size 8192
    --sglang-max-prefill-tokens 8192
    --sglang-max-running-requests 32
    --sglang-watchdog-timeout 3000000
    # PLE n-gram table stays in host memory; frees ~10 GB/card for KV cache.
    --sglang-ple-offload-embedding
    # Hybrid GDN prefix cache: extra_buffer is the strategy the server script uses.
    --sglang-mamba-radix-cache-strategy extra_buffer
    --sglang-mamba-track-interval 64000000
    --sglang-max-mamba-cache-size 160
    --sglang-cuda-graph-max-bs-decode 32

    --sglang-mm-attention-backend fa3
    --sglang-mm-enable-dp-encoder
)

TRACKING_ARGS=(
    --tb-experiment-name qwen3.8-flash-next
    --use-wandb
    --wandb-project qwen3.8-flash-next
    --wandb-group qwen3.8-flash-next-klx-${now}
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

EXTRA_ENV_VARS_JSON="\"SGLANG_ENABLE_QWEN4_QSA\": \"1\",
    \"SGLANG_ENABLE_QWEN4_QSA_SPARSE_PREFILL\": \"${SGLANG_ENABLE_QWEN4_QSA_SPARSE_PREFILL:-1}\",
    \"SGLANG_QSA_USE_GATHER_MULTISTREAM\": \"${SGLANG_QSA_USE_GATHER_MULTISTREAM:-0}\",
    \"XSGL_MTP_QUANTIZE\": \"0\",
    \"XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL\": \"True\""

source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
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

    # --debug-rollout-only \
    # --save-debug-rollout-data /data1/zhujinlei/qwen38/rollout_data/{rollout_id}.pt \
    # --debug-train-only \
    # --load-debug-rollout-data /data1/zhujinlei/qwen38/rollout_data/{rollout_id}.pt \