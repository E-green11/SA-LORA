
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict

import datasets
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint

# Import LoraPlus components
current_dir = os.path.dirname(__file__)
loraplus_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(loraplus_dir)

from salora import saloraTrainer
from arguments import TrainingArguments, ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )

    set_seed(training_args.seed)

    # Load dataset
    if data_args.train_file is not None:
        # Load from local file
        extension = data_args.train_file.split(".")[-1]
        data_files = {"train": data_args.train_file}
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Load from Hugging Face Hub (e.g., "tatsu-lab/alpaca")
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Default to Alpaca dataset
        raw_datasets = load_dataset(
            "tatsu-lab/alpaca",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    
    # If no validation set, create one from train
    if "validation" not in raw_datasets:
        raw_datasets = raw_datasets["train"].train_test_split(test_size=0.02, seed=42)
        raw_datasets["validation"] = raw_datasets.pop("test")

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
    )

    # LLaMA tokenizer doesn't have pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine torch dtype
    torch_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        torch_dtype=torch_dtype,
    )

    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Setup LoRA
    if hasattr(training_args, "use_lora") and training_args.use_lora:
        # Default target modules for LLaMA
        target_modules = ["q_proj", "v_proj"]
        
        if hasattr(training_args, "target_modules") and training_args.target_modules:
            if isinstance(training_args.target_modules, str):
                target_modules = [t.strip() for t in training_args.target_modules.split(",")]
            else:
                target_modules = training_args.target_modules

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=getattr(training_args, "lora_rank", 8),
            lora_alpha=getattr(training_args, "lora_alpha", 16),
            lora_dropout=getattr(training_args, "lora_dropout", 0.1),
            target_modules=target_modules,
        )

        model = get_peft_model(model, peft_config)
        logger.info("Training using LoRA!")
        model.print_trainable_parameters()

    # Preprocessing function
    def preprocess_function(examples):
        """Format examples into Alpaca prompt template."""
        sources = []
        targets = []
        
        for instruction, input_text, output in zip(
            examples.get("instruction", []),
            examples.get("input", []),
            examples.get("output", [])
        ):
            if input_text and input_text.strip():
                prompt = ALPACA_PROMPT_TEMPLATE.format(
                    instruction=instruction,
                    input=input_text,
                    output=""
                )
            else:
                prompt = ALPACA_PROMPT_NO_INPUT.format(
                    instruction=instruction,
                    output=""
                )
            sources.append(prompt)
            targets.append(output + tokenizer.eos_token)

        # Tokenize
        model_inputs = tokenizer(
            sources,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding=False,
        )

        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding=False,
        )

        # Combine source and target
        input_ids = []
        attention_masks = []
        label_ids = []

        for source_ids, target_ids in zip(model_inputs["input_ids"], labels["input_ids"]):
            # Combine input and output
            combined_ids = source_ids + target_ids
            
            # Truncate if too long
            if len(combined_ids) > data_args.max_seq_length:
                combined_ids = combined_ids[:data_args.max_seq_length]
            
            # Create attention mask
            attention_mask = [1] * len(combined_ids)
            
            # Create labels: -100 for source tokens (don't compute loss), actual ids for target
            source_len = len(source_ids)
            if source_len > data_args.max_seq_length:
                source_len = data_args.max_seq_length
            
            label = [-100] * source_len + target_ids
            if len(label) > data_args.max_seq_length:
                label = label[:data_args.max_seq_length]
            
            input_ids.append(combined_ids)
            attention_masks.append(attention_mask)
            label_ids.append(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": label_ids,
        }

    # Process datasets
    column_names = raw_datasets["train"].column_names

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = raw_datasets["validation"].map(
                preprocess_function,
                batched=True,
                num_proc=4,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # Initialize trainer
    trainer = saloraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

