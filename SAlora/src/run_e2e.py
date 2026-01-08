
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Import LoraPlus components
# This script assumes it is run from glue/scripts/ while run_e2e.py is located in glue/src/
# lora_plus.py is located two levels up (project root)
current_dir = os.path.dirname(__file__)
loraplus_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(loraplus_dir)

from lora_plus import LoraPlusTrainer
# Import TrainingArguments from local arguments.py which inherits from LoraPlusTrainingArguments and adds LoRA specific args
from arguments import TrainingArguments, ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

# Remove duplicate definitions of ModelArguments and DataTrainingArguments as we import them now

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
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

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
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
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Load dataset
    if data_args.train_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        # E2E dataset has 'meaning_representation' and 'human_reference'
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
    )
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    
    # Enable gradient checkpointing for memory efficiency (must be before LoRA setup)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

    # Setup LoRA
    if hasattr(training_args, "use_lora") and training_args.use_lora:
        target_modules = ["c_attn"] # Default for GPT-2
        # If specific target modules are provided in args (need to check LoraPlusTrainingArguments definition)
        # It seems LoraPlusTrainingArguments inherits from TrainingArguments, which doesn't have target_modules by default in transformers
        # BUT, run_glue.py assumes training_args has target_modules. 
        # Let's assume we pass it via some way or manually handle it here if it's not in LoraPlusTrainingArguments.
        # Actually LoraPlusTrainingArguments in lora_plus.py doesn't explicitly define target_modules, 
        # so it must be added dynamically or via the base TrainingArguments if updated.
        # For safety, we'll parse it manually if needed, or trust HfArgumentParser to pick up extra args if defined.
        
        # Checking how run_glue.py does it: training_args.target_modules
        # We will assume it's available or we use default.
        if hasattr(training_args, "target_modules") and training_args.target_modules:
             if isinstance(training_args.target_modules, str):
                target_modules = [t.strip() for t in training_args.target_modules.split(",")]
             else:
                target_modules = training_args.target_modules
        
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", 
            inference_mode=False, 
            r=getattr(training_args, "lora_rank", 8), 
            lora_alpha=getattr(training_args, "lora_alpha", 8), 
            lora_dropout=getattr(training_args, "lora_dropout", 0.1),
            target_modules=target_modules
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Preprocessing
    column_names = raw_datasets["train"].column_names
    if "mr" in column_names and "ref" in column_names:
        mr_col = "mr"
        ref_col = "ref"
    else:
        mr_col = "meaning_representation"
        ref_col = "human_reference"

    def preprocess_function(examples):
        # E2E formatting: "MR <|endoftext|> REF <|endoftext|>"
        # Or for conditional generation: "MR \n REF"
        # Standard GPT-2 approach: concat input and target.
        
        inputs = examples[mr_col]
        targets = examples[ref_col]
        
        model_inputs = []
        for inp, tgt in zip(inputs, targets):
            # Simple formatting: Input: MR \n Output: Ref <eos>
            # Or just: MR <sep> Ref <eos>
            # Let's use a format: "Meaning Representation: {mr}\nReference: {ref}<|endoftext|>"
            text = f"Meaning Representation: {inp}\nReference: {tgt}{tokenizer.eos_token}"
            model_inputs.append(text)
            
        tokenized = tokenizer(
            model_inputs, 
            truncation=True, 
            max_length=data_args.max_seq_length,
            padding="max_length"
        )
        
        # For Causal LM, labels are same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        # Mask out the loss for the input part (optional, but good for performance)
        # Here we just do standard CLM training on the whole sequence for simplicity
        # To be more precise, we could set labels to -100 for the MR part.
        
        return tokenized

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=4,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=4,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Trainer
    trainer = LoraPlusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

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

