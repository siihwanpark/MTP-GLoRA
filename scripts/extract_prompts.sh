# !/bin/bash

DATASET=${1:-am_qwen3}
OUT_PATH=path/to/save
if [ $DATASET == "openthoughts" ]; then
    OUT_FILE=openthoughts114k_prompts.jsonl
elif [ $DATASET == "am_qwen3" ]; then
    OUT_FILE=am_qwen3_distill_prompts.jsonl
else
    echo "Invalid dataset: $DATASET. Available datasets: openthoughts, am_qwen3"
    exit 1
fi

EXTRA_ARGS=""
if [ $DATASET == "openthoughts" ]; then
    EXTRA_ARGS="--code_mix '{\"stdin\":0.6,\"solve\":0.4,\"none\":0.0}'\
                --max_samples 50000"
elif [ $DATASET == "am_qwen3" ]; then
    EXTRA_ARGS="--repo a-m-team/AM-Qwen3-Distilled\
                --total_samples 122880\
                --weights '{\"chat\":0.0,\"math\":0.536,\"code\":0.311,\"science\":0.153,\"if\":0.0}'\
                --num_workers 8"
fi

python prepare_data/extract_prompts.py \
  --source $DATASET \
  --out_path $OUT_PATH/$OUT_FILE \
  --report_json $OUT_PATH/${OUT_FILE}.stats.json \
  $EXTRA_ARGS