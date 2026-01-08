#!/bin/bash

# MT-Bench GPT-4 Judging for LLaMA 7B + SA-LoRA

# Note: generate_answers.py saves to parent directory of LORA_PATH
INPUT_FILE=output/llama_7b_sa_lora_alpaca/mtbench_answers.json
# Run judging
python mtbench/judge_answers.py \
    --input-file $INPUT_FILE

echo ""
echo "MT-Bench judging completed!"

