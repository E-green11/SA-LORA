#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=true

# MT-Bench evaluation for LLaMA 7B + SA-LoRA

# Base model path
BASE_MODEL=huggyllama/llama-7b

# LoRA adapter path (trained model)
LORA_PATH=output/llama_7b_sa_lora_alpaca

# Run evaluation
python mtbench/generate_answers.py \
    --base-model $BASE_MODEL \
    --lora-path $LORA_PATH \
    --output-file mtbench_answers.json

echo "MT-Bench answer generation completed!"
echo "Answers saved to: output/llama_7b_sa_lora_alpaca/mtbench_answers.json"

