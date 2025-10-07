#!/bin/bash

MODEL_PATH="path/to/model"
DATA_PATH="path/to/data"
SAVE_DIR="path/to/save"

ts() { date '+%F %T'; }

has_checkpoint() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  if [[ -e "$dir/latest.json" ]] || compgen -G "$dir/*.pt" >/dev/null || compgen -G "$dir/*.safetensors" >/dev/null || compgen -G "$dir/*.bin" >/dev/null; then
    return 0
  fi
  return 1
}

cleanup() {
  echo "[$(ts)] INFO: Caught signal. Terminating child jobs..."
  jobs -pr | xargs -r kill
  exit 0
}
trap cleanup INT TERM

extra_args=()
if has_checkpoint "$SAVE_DIR"; then
    extra_args+=(--resume --checkpoint_dir "$SAVE_DIR")
fi

torchrun --standalone --nproc_per_node=8 -m mtp_glora.train\
    --model_path $MODEL_PATH\
    --train_data_path $DATA_PATH\
    --save_dir $SAVE_DIR\
    --report_to tensorboard\
    --fuse_weights\
    --lr 2e-4 --warmup_steps 5000 --max_steps 50000 --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05\
    --save_limit 10 --save_steps 1000\
    --chunk_size 5120 --min_chunk_size 1024\
    "${extra_args[@]}"

echo "[$(ts)] INFO: Training completed"
exit 0