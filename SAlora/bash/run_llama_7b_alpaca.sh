#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# Environment setup
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4
export WANDB_DISABLED=true

# Experiment configuration
model_name=huggyllama/llama-7b
model_short=llama_7b
task=alpaca
exp_name=${model_short}_sa_lora_${task}
lr=2e-4
lr_ratio=25

# SA-LoRA configuration
use_sa_lora=true
sa_metric=stable_rank
sa_min_mult=0.4
sa_max_mult=3.2
sa_power=2.5
sa_apply_to=b
sa_warmup_steps=500
sa_normalize_mean=True
sa_online_ema=true
sa_ema_beta=0.9
sa_ema_update_every=100
sa_depth_prior=true
sa_depth_prior_weight=0.1

# Execute command
python src/run_instruction_tuning.py \
  --model_name_or_path $model_name \
  --dataset_name "yahma/alpaca-cleaned" \
  --use_lora \
  --target_modules "q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj" \
  --lora_rank 64 \
  --lora_alpha 64 \
  --loraplus_lr_ratio $lr_ratio \
  $( [ "$use_sa_lora" = true ] && echo --use_sa_lora ) \
  --sa_metric $sa_metric \
  --sa_min_mult $sa_min_mult \
  --sa_max_mult $sa_max_mult \
  --sa_power $sa_power \
  --sa_apply_to $sa_apply_to \
  --sa_warmup_steps $sa_warmup_steps \
  --sa_normalize_mean $sa_normalize_mean \
  $( [ "$sa_online_ema" = true ] && echo --sa_online_ema ) \
  --sa_ema_beta $sa_ema_beta \
  --sa_ema_update_every $sa_ema_update_every \
  $( [ "$sa_depth_prior" = true ] && echo --sa_depth_prior ) \
  --sa_depth_prior_weight $sa_depth_prior_weight \
  --sa_grad_calibrate_steps 200 \
  --sa_grad_power 1.5 \
  --sa_grad_blend 0.2 \
  --do_train \
  --do_eval \
  --bf16 \
  --gradient_checkpointing \
  --max_seq_length 1024 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 14 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate $lr \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_steps 500 \
  --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
  --logging_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
  --overwrite_output_dir \
  --save_total_limit 10 \
  --save_safetensors True \
  --report_to tensorboard

