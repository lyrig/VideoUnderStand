#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
NUM_PROCESSES=${NUM_PROCESSES:-2}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
MODEL_PATH=${MODEL_PATH:-/mnt/dhwfile/raise/user/panjiabao/huxiaobin/shy/ckpt/model/Qwen2.5-VL-3B-Instruct}
TRAIN_JSONL=${TRAIN_JSONL:-/mnt/dhwfile/raise/user/panjiabao/huxiaobin/shy/data/Video-R1-data/Video-R1-COT-165k.json}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/dhwfile/raise/user/panjiabao/huxiaobin/shy/ckpt/VideoUnderStand/stage1_zero2}
CONFIG_PATH=${CONFIG_PATH:-configs/vismem_qwen25vl7b.yaml}
EPOCHS=${EPOCHS:-1}

export CUDA_VISIBLE_DEVICES

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --use_deepspeed \
  --zero_stage 2 \
  -m main.cli.train_stage1_zero2 \
  --config "${CONFIG_PATH}" \
  --model_name_or_path "${MODEL_PATH}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}"
