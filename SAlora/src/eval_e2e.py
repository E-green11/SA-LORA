#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import os
import sys
import csv
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add project root to path
current_dir = os.path.dirname(__file__)
loraplus_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(loraplus_dir)

# Try to import evaluation metrics
try:
    import evaluate
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False
    print("Warning: 'evaluate' library not found. Install with: pip install evaluate")

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.nist_score import corpus_nist
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("Warning: 'nltk' not found. Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("Warning: 'rouge_score' not found. Install with: pip install rouge-score")


def load_test_data(test_file):
    """Load test data and group references by MR."""
    mr_to_refs = defaultdict(list)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mr = row['mr']
            ref = row['ref']
            mr_to_refs[mr].append(ref)
    
    return mr_to_refs


def load_model(model_path, base_model_name=None):
    """Load the trained model (with LoRA if applicable)."""
    # Check if this is a LoRA model
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        # This is a LoRA model
        print(f"Loading LoRA model from {model_path}")
        
        # Get base model name from adapter config
        if base_model_name is None:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "gpt2-medium")
        
        # Load base model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
    else:
        # Standard model
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
    # Format input the same way as training
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
    
    # Decode and extract the generated reference
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated reference part
    if "Reference:" in generated:
        generated = generated.split("Reference:")[-1].strip()
    
    # Clean up any trailing special tokens or extra content
    generated = generated.split("\n")[0].strip()
    
    return generated


def compute_bleu_nist(predictions, references_list):
    """Compute BLEU and NIST scores using NLTK."""
    if not HAS_NLTK:
        return {"bleu": None, "nist": None}
    
    # Tokenize
    tokenized_preds = [pred.lower().split() for pred in predictions]
    tokenized_refs = [[ref.lower().split() for ref in refs] for refs in references_list]
    
    # BLEU
    smoothing = SmoothingFunction().method1
    bleu = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothing)
    
    # NIST (requires at least 5-grams to work properly)
    try:
        nist = corpus_nist(tokenized_refs, tokenized_preds, n=5)
    except Exception as e:
        print(f"NIST computation error: {e}")
        nist = 0.0
    
    return {"bleu": bleu * 100, "nist": nist}


def compute_rouge_l(predictions, references_list):
    """Compute ROUGE-L score."""
    if not HAS_ROUGE:
        return {"rouge_l": None}
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    scores = []
    for pred, refs in zip(predictions, references_list):
        # Take max score across all references
        ref_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref in refs]
        scores.append(max(ref_scores))
    
    return {"rouge_l": sum(scores) / len(scores) * 100}


def compute_meteor(predictions, references_list):
    """Compute METEOR score using evaluate library."""
    if not HAS_EVALUATE:
        return {"meteor": None}
    
    try:
        meteor = evaluate.load("meteor")
        
        # For METEOR, we use the first reference for simplicity
        # (or can compute average across all references)
        flat_refs = [refs[0] for refs in references_list]
        
        results = meteor.compute(predictions=predictions, references=flat_refs)
        return {"meteor": results['meteor'] * 100}
    except Exception as e:
        print(f"METEOR computation error: {e}")
        return {"meteor": None}


def compute_cider(predictions, references_list):
    """
    Compute CIDEr score.
    CIDEr requires special handling - using a simplified version here.
    For official results, use the e2e-metrics package.
    """
    # Try to use the cider package if available
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # Format for CIDEr scorer
        gts = {i: refs for i, refs in enumerate(references_list)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return {"cider": score * 100}
    except ImportError:
        pass
    
    # Fallback: simplified TF-IDF based CIDEr approximation
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        scores = []
        for pred, refs in zip(predictions, references_list):
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer()
            all_texts = [pred] + refs
            try:
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                # Compute similarity between prediction and each reference
                pred_vec = tfidf_matrix[0:1]
                ref_vecs = tfidf_matrix[1:]
                similarities = cosine_similarity(pred_vec, ref_vecs)[0]
                scores.append(np.mean(similarities))
            except:
                scores.append(0.0)
        
        return {"cider": np.mean(scores) * 100}
    except ImportError:
        return {"cider": None}


def main():
    parser = argparse.ArgumentParser(description="E2E NLG Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="Base model name (for LoRA models)")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test CSV file with MR and references")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save evaluation results JSON")
    parser.add_argument("--predictions_file", type=str, default=None,
                        help="Path to save generated predictions")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set output file path if not specified
    if args.output_file is None:
        args.output_file = os.path.join(args.model_path, "e2e_eval_results.json")
    if args.predictions_file is None:
        args.predictions_file = os.path.join(args.model_path, "e2e_predictions.txt")
    
    print("=" * 60)
    print("E2E NLG Benchmark Evaluation")
    print("=" * 60)
    
    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    mr_to_refs = load_test_data(args.test_file)
    print(f"Loaded {len(mr_to_refs)} unique MRs")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, args.base_model_name)
    
    # Generate predictions
    print("\nGenerating predictions...")
    mrs = list(mr_to_refs.keys())
    references_list = [mr_to_refs[mr] for mr in mrs]
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
    print(f"\nSaving predictions to {args.predictions_file}...")
    with open(args.predictions_file, 'w', encoding='utf-8') as f:
        for mr, pred in zip(mrs, predictions):
            f.write(f"MR: {mr}\n")
            f.write(f"Generated: {pred}\n")
            f.write("-" * 50 + "\n")
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    
    results = {}
    
    # BLEU & NIST
    bleu_nist = compute_bleu_nist(predictions, references_list)
    results.update(bleu_nist)
    print(f"  BLEU: {bleu_nist['bleu']:.2f}" if bleu_nist['bleu'] else "  BLEU: N/A")
    print(f"  NIST: {bleu_nist['nist']:.2f}" if bleu_nist['nist'] else "  NIST: N/A")
    
    # METEOR
    meteor = compute_meteor(predictions, references_list)
    results.update(meteor)
    print(f"  METEOR: {meteor['meteor']:.2f}" if meteor['meteor'] else "  METEOR: N/A")
    
    # ROUGE-L
    rouge = compute_rouge_l(predictions, references_list)
    results.update(rouge)
    print(f"  ROUGE-L: {rouge['rouge_l']:.2f}" if rouge['rouge_l'] else "  ROUGE-L: N/A")
    
    # CIDEr
    cider = compute_cider(predictions, references_list)
    results.update(cider)
    print(f"  CIDEr: {cider['cider']:.2f}" if cider['cider'] else "  CIDEr: N/A")
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print("\nResults Summary:")
    print(f"  BLEU:    {results.get('bleu', 'N/A'):.2f}" if results.get('bleu') else "  BLEU:    N/A")
    print(f"  NIST:    {results.get('nist', 'N/A'):.2f}" if results.get('nist') else "  NIST:    N/A")
    print(f"  METEOR:  {results.get('meteor', 'N/A'):.2f}" if results.get('meteor') else "  METEOR:  N/A")
    print(f"  ROUGE-L: {results.get('rouge_l', 'N/A'):.2f}" if results.get('rouge_l') else "  ROUGE-L: N/A")
    print(f"  CIDEr:   {results.get('cider', 'N/A'):.2f}" if results.get('cider') else "  CIDEr:   N/A")


if __name__ == "__main__":
    main()

