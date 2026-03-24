#!/usr/bin/env python3
"""
Evaluate trained ablation models across all conditions.
"""

import json
import torch
import argparse
import re
from pathlib import Path
from typing import Literal

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm


InputFormat = Literal["full", "response-only", "redacted"]


def extract_question(prompt: str) -> str:
    """Extract just the question, removing hint preambles."""
    patterns = [
        r"I recall that .+? might be the answer\.\s*",
        r"I think the answer is .+?\.\s*",
        r"Someone told me .+?\.\s*",
    ]
    result = prompt
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result.strip()


def format_input(example: dict, input_format: InputFormat) -> str:
    """Format input based on ablation condition."""
    if input_format == "full":
        return f"{example['prompt']}\n\nResponse: {example['generated_text']}"
    elif input_format == "response-only":
        return example['generated_text'].strip()
    elif input_format == "redacted":
        question = extract_question(example['prompt'])
        return f"{question}\n\nResponse: {example['generated_text']}"
    else:
        raise ValueError(f"Unknown input format: {input_format}")


def load_test_data(data_path: Path, val_split: float = 0.15):
    """Load validation split (same as training script)."""
    import random
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    
    random.seed(42)
    random.shuffle(examples)
    
    split_idx = int(len(examples) * (1 - val_split))
    return examples[split_idx:]  # Return validation set


def evaluate_model(model_dir: Path, test_examples: list[dict], input_format: InputFormat, device: str = "mps"):
    """Evaluate a single model."""
    # Load config to get base model
    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        print(f"  No training_config.json in {model_dir}, skipping")
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    base_model = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    trained_format = config.get("input_format", "full")
    
    print(f"  Loading {model_dir.name} (trained on: {trained_format})")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for ex in tqdm(test_examples, desc="Evaluating", leave=False):
            text = format_input(ex, input_format)
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
            
            all_probs.append(probs)
            all_labels.append(1 if ex['label'] == 'unfaithful' else 0)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    
    auroc = roc_auc_score(all_labels, all_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='binary', zero_division=0
    )
    accuracy = (preds == all_labels).mean()
    
    return {
        "model": model_dir.name,
        "trained_format": trained_format,
        "eval_format": input_format,
        "auroc": round(auroc, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--models-dir", type=str, required=True, help="Directory containing model subdirs")
    parser.add_argument("--eval-format", type=str, choices=["full", "response-only", "redacted", "all"],
                        default="all", help="Input format for evaluation")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    test_examples = load_test_data(Path(args.test_data))
    print(f"Loaded {len(test_examples)} test examples")
    
    # Find all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and (d / "training_config.json").exists()]
    print(f"Found {len(model_dirs)} models: {[d.name for d in model_dirs]}")
    
    eval_formats = ["full", "response-only", "redacted"] if args.eval_format == "all" else [args.eval_format]
    
    results = []
    for model_dir in model_dirs:
        for fmt in eval_formats:
            print(f"\nEvaluating {model_dir.name} with {fmt} format:")
            result = evaluate_model(model_dir, test_examples, fmt, args.device)
            if result:
                results.append(result)
                print(f"  AUROC: {result['auroc']:.4f}, F1: {result['f1']:.4f}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Trained On':<15} {'Eval Format':<15} {'AUROC':<8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x['model'], x['eval_format'])):
        print(f"{r['model']:<25} {r['trained_format']:<15} {r['eval_format']:<15} {r['auroc']:<8.4f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
