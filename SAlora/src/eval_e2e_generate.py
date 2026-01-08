#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate predictions for E2E benchmark in official format.
One prediction per line, in the same order as the MRs.
"""

import argparse
import json
import os
import sys
import csv
from collections import OrderedDict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add project root to path
current_dir = os.path.dirname(__file__)
loraplus_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(loraplus_dir)


def load_test_data(test_file):
    """Load test data - keep order and get unique MRs."""
    mrs = []
    mr_to_refs = OrderedDict()
    
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mr = row['mr']
            ref = row['ref']
            if mr not in mr_to_refs:
                mr_to_refs[mr] = []
                mrs.append(mr)
            mr_to_refs[mr].append(ref)
    
    return mrs, mr_to_refs


def load_model(model_path, base_model_name=None):
    """Load the trained model (with LoRA if applicable)."""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        print(f"Loading LoRA model from {model_path}")
        
        if base_model_name is None:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "gpt2-medium")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, mr, max_new_tokens=100, num_beams=5, device="cuda"):
    """Generate text for a given meaning representation."""
    prompt = f"Meaning Representation: {mr}\nReference:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Reference:" in generated:
        generated = generated.split("Reference:")[-1].strip()
    
    generated = generated.split("\n")[0].strip()
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate E2E predictions")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    print("Loading test data...")
    mrs, mr_to_refs = load_test_data(args.test_file)
    print(f"Loaded {len(mrs)} unique MRs")
    
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.base_model_name)
    
    print("Generating predictions...")
    predictions = []
    for mr in tqdm(mrs, desc="Generating"):
        pred = generate_text(
            model, tokenizer, mr,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            device=args.device
        )
        predictions.append(pred)
    
    # Save predictions - one per line
    print(f"Saving predictions to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + "\n")
    
    print("Done!")


if __name__ == "__main__":
    main()

