#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E2E NLG Benchmark Official Evaluation
Uses pycocoevalcap to compute official metrics: BLEU, NIST, METEOR, ROUGE-L, CIDEr.

Install dependencies:
    pip install pycocoevalcap nltk

Usage:
    python eval_e2e_official.py \
        --model_path output/gpt2-medium_lora_e2e_sa/lr-2e-4_ratio-25 \
        --base_model_name gpt2-medium \
        --test_file data/e2e_nlg/testset_w_refs.csv
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

# Setup NLTK
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

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


def compute_metrics_pycocoevalcap(predictions, references_list):
    """
    Compute metrics using pycocoevalcap (same as official e2e-metrics).
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    
    # NIST is not in pycocoevalcap, compute separately
    from nltk.translate.nist_score import corpus_nist
    
    # Format for pycocoevalcap: dict of {id: [sentences]}
    gts = {}  # ground truth (references)
    res = {}  # results (predictions)
    
    for i, (pred, refs) in enumerate(zip(predictions, references_list)):
        gts[i] = refs
        res[i] = [pred]
    
    results = {}
    
    # BLEU
    print("  Computing BLEU...")
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    results['BLEU'] = bleu_scores[3] * 100  # BLEU-4
    
    # NIST (using NLTK)
    print("  Computing NIST...")
    try:
        tokenized_preds = [pred.lower().split() for pred in predictions]
        tokenized_refs = [[ref.lower().split() for ref in refs] for refs in references_list]
        nist_score = corpus_nist(tokenized_refs, tokenized_preds, n=5)
        results['NIST'] = nist_score
    except Exception as e:
        print(f"    NIST error: {e}")
        results['NIST'] = 0.0
    
    # METEOR
    print("  Computing METEOR...")
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts, res)
    results['METEOR'] = meteor_score * 100
    
    # ROUGE-L
    print("  Computing ROUGE-L...")
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    results['ROUGE-L'] = rouge_score * 100
    
    # CIDEr
    print("  Computing CIDEr...")
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    results['CIDEr'] = cider_score  # CIDEr is already in the right scale
    
    return results


def main():
    parser = argparse.ArgumentParser(description="E2E Official Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, load from existing predictions file")
    parser.add_argument("--predictions_file", type=str, default=None,
                        help="Load predictions from this file (one per line)")
    
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = os.path.join(args.model_path, "e2e_official_results.json")
    
    print("=" * 60)
    print("E2E NLG Official Benchmark Evaluation")
    print("=" * 60)
    
    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    mrs, mr_to_refs = load_test_data(args.test_file)
    references_list = [mr_to_refs[mr] for mr in mrs]
    print(f"Loaded {len(mrs)} unique MRs")
    
    # Generate or load predictions
    if args.skip_generation and args.predictions_file:
        print(f"\nLoading predictions from {args.predictions_file}...")
        with open(args.predictions_file, 'r', encoding='utf-8') as f:
            predictions = [line.strip() for line in f]
    else:
        print(f"\nLoading model from {args.model_path}...")
        model, tokenizer = load_model(args.model_path, args.base_model_name)
        
        print("\nGenerating predictions...")
        predictions = []
        for mr in tqdm(mrs, desc="Generating"):
            pred = generate_text(
                model, tokenizer, mr,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                device=args.device
            )
            predictions.append(pred)
        
        # Save predictions
        pred_file = os.path.join(args.model_path, "predictions_official.txt")
        with open(pred_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred + "\n")
        print(f"Predictions saved to {pred_file}")
    
    # Compute metrics
    print("\nComputing official metrics...")
    results = compute_metrics_pycocoevalcap(predictions, references_list)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results (Official Metrics)")
    print("=" * 60)
    print(f"  BLEU:    {results['BLEU']:.2f}")
    print(f"  NIST:    {results['NIST']:.2f}")
    print(f"  METEOR:  {results['METEOR']:.2f}")
    print(f"  ROUGE-L: {results['ROUGE-L']:.2f}")
    print(f"  CIDEr:   {results['CIDEr']:.2f}")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

