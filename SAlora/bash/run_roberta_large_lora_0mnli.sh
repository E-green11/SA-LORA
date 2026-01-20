#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# Environment setup
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=6

# Experiment configuration
task=mnli
exp_name=roberta_large_lora_$task
lr=2e-4           # ###########################
use_sa_lora=true         # SA-LoRA switch
sa_metric=stable_rank    # stable_rank | spectral_entropy | cond
sa_min_mult=0.4
sa_max_mult=3.2
sa_power=2.5
sa_apply_to=b            # a | b | both (apply to 'b' first for stability)
sa_warmup_steps=2000
sa_normalize_mean=True
# New SA options
sa_online_ema=true
sa_ema_beta=0.9
sa_ema_update_every=100
sa_depth_prior=true
sa_depth_prior_weight=0.1

# Execute command
python src/run_glue.py \
  --model_name_or_path roberta-large \
  --task_name $task \
  --use_lora \
  --target_modules "query, value" \
  --do_train \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size   \
  --max_seq_length   \
  --eval_steps   \
  --save_steps   \
  --logging_steps   \
  --num_train_epochs   \
  --learning_rate $lr \
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
  --sa_grad_calibrate_steps 1000 \
  --sa_grad_power   \
  --sa_grad_blend   \
  --lora_rank   \
  --lora_alpha   \
  --fp16 \
  --lr_scheduler_type 'linear' \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --adam_epsilon 1e-8 \
  --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
  --logging_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
  --evaluation_strategy steps \
  --save_strategy steps \
  --load_best_model_at_end \
  --metric_for_best_model accuracy \
  --greater_is_better True \
  --report_to tensorboard \
  --keep_checkpoints eval \
  --overwrite_output_dir \
  --ignore_mismatched_sizes \
  --save_total_limit 40 \
  --save_on_each_node \
  --save_safetensors True
