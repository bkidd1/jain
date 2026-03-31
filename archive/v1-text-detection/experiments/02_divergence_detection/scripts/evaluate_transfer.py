#!/usr/bin/env python3
"""
Evaluate cross-model transfer: train detector on Model A, test on Model B.

This is the key experiment - can we detect unfaithful reasoning in models
we've never seen the internals of?
"""

import json
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoTokenizer
from peft import PeftModel, AutoPeftModelForSequenceClassification
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import numpy as np


def load_examples(path: Path) -> list[dict]:
    """Load extraction examples."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def predict(model, tokenizer, examples: list[dict], device: str, batch_size: int = 1):
    """Get predictions for examples."""
    model.eval()
    
    all_probs = []
    all_labels = []
    
    # Use batch_size=1 to avoid padding issues
    for i in tqdm(range(0, len(examples), batch_size), desc="Predicting"):
        batch = examples[i:i+batch_size]
        
        # Format inputs
        texts = [f"{ex['prompt']}\n\nResponse: {ex['generated_text']}" for ex in batch]
        labels = [1 if ex['label'] == 'unfaithful' else 0 for ex in batch]
        
        # Tokenize
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(
                input_ids=encoded['input_ids'].to(device),
                attention_mask=encoded['attention_mask'].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        
        all_probs.extend(probs.tolist())
        all_labels.extend(labels)
    
    return np.array(all_probs), np.array(all_labels)


def evaluate(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    """Compute evaluation metrics."""
    preds = (probs > threshold).astype(int)
    
    # Core metrics
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = 0.5
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    accuracy = (preds == labels).mean()
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained detector")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test extractions")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    test_path = Path(args.test_data)
    
    print(f"Loading model from {model_dir}")
    print(f"Testing on {test_path}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    ).to(args.device)
    
    # Load test data
    examples = load_examples(test_path)
    print(f"Loaded {len(examples)} test examples")
    
    # Count labels
    n_unfaithful = sum(1 for ex in examples if ex['label'] == 'unfaithful')
    print(f"Unfaithful: {n_unfaithful} ({100*n_unfaithful/len(examples):.1f}%)")
    
    # Get predictions
    probs, labels = predict(model, tokenizer, examples, args.device)
    
    # Evaluate
    results = evaluate(probs, labels)
    
    # Print results
    print("\n" + "="*60)
    print("Transfer Evaluation Results")
    print("="*60)
    print(f"AUROC:     {results['auroc']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1:        {results['f1']:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {results['true_positives']:4d}  FP: {results['false_positives']:4d}")
    print(f"  FN: {results['false_negatives']:4d}  TN: {results['true_negatives']:4d}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = test_path.parent / f"transfer_results_{test_path.stem}.json"
    
    output_data = {
        "model_dir": str(model_dir),
        "test_data": str(test_path),
        "n_examples": len(examples),
        "n_unfaithful": n_unfaithful,
        "metrics": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
