#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ZERO_STAGE=${ZERO_STAGE:-2}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/deepspeed_smoke}
CONFIG_PATH=${CONFIG_PATH:-configs/vismem_qwen25vl7b.yaml}
STEPS=${STEPS:-1}
FORMER_BACKEND=${FORMER_BACKEND:-tiny_transformer}

export CUDA_VISIBLE_DEVICES

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --use_deepspeed \
  --zero_stage "${ZERO_STAGE}" \
  -m main.cli.test_deepspeed_smoke \
  --config "${CONFIG_PATH}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --steps "${STEPS}" \
  --zero_stage "${ZERO_STAGE}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --former_backend "${FORMER_BACKEND}"
