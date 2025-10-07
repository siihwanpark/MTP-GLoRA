#!/bin/bash

# TORCH_CUDA_ARCH_LIST = 8.0 for A100, 9.0 for H100
export TORCH_CUDA_ARCH_LIST=9.0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASET=${1:-am_qwen3}
MODEL_PATH=path/to/model
INPUT_JSONL=path/to/data
OUTPUT_DIR=path/to/save

python prepare_data/run_vllm.py \
    --model $MODEL_PATH \
    --tp_size 8 \
    --input_jsonl $INPUT_JSONL \
    --output_dir $OUTPUT_DIR \
    --max_gen_len 32768 \
    --max_model_len 34816 \
    --temperature 0.6 --top_p 0.95 --top_k 20