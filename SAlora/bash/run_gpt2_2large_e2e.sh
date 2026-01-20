#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# Environment setup
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4

# Experiment configuration
model_name=gpt2-large
task=e2e
exp_name=${model_name}_lora_${task}_sa
lr=2e-4
use_sa_lora=true
sa_metric=stable_rank
sa_min_mult=0.4
sa_max_mult=3.2
sa_power=2.5
sa_apply_to=b
sa_warmup_steps=2000
sa_normalize_mean=True

# Execute command
python src/run_e2e.py \
  --model_name_or_path $model_name \
  --train_file data/e2e_nlg/trainset.csv \
  --validation_file data/e2e_nlg/devset.csv \
  --use_lora \
  --use_sa_lora \
  --sa_metric $sa_metric \
  --sa_min_mult $sa_min_mult \
  --sa_max_mult $sa_max_mult \
  --sa_power $sa_power \
  --sa_apply_to $sa_apply_to \
  --sa_warmup_steps $sa_warmup_steps \
  --sa_normalize_mean $sa_normalize_mean \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules "c_attn" \
  --do_train \
  --do_eval \
  # --bf16 \
  # --gradient_checkpointing \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 10 \
  --num_train_epochs 8 \
  --learning_rate $lr \
  --optim adamw_torch \
  --lr_scheduler_type 'linear' \
  --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
  --overwrite_output_dir \
  --save_total_limit 20 \
  --save_safetensors True

