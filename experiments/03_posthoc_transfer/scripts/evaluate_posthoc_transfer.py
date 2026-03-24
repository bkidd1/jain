#!/usr/bin/env python3
"""
Evaluate whether our hint-detector transfers to post-hoc rationalization.

This is the key test: Can a detector trained on "hint-informed reasoning"
detect "post-hoc rationalization" without any retraining?

If yes → there's a common structural signature of unfaithful reasoning.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_fscore_support,
    confusion_matrix,
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
    all_conditions = []
    
    for i in tqdm(range(0, len(examples), batch_size), desc="Predicting"):
        batch = examples[i:i+batch_size]
        
        # Format inputs - same format as hint detector training
        texts = [f"{ex['prompt']}\n\nResponse: {ex['generated_text']}" for ex in batch]
        labels = [1 if ex['label'] == 'unfaithful' else 0 for ex in batch]
        conditions = [ex['condition'] for ex in batch]
        
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
        all_conditions.extend(conditions)
    
    return np.array(all_probs), np.array(all_labels), all_conditions


def evaluate(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    """Compute evaluation metrics."""
    preds = (probs > threshold).astype(int)
    
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = 0.5
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    accuracy = (preds == labels).mean()
    
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
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained hint detector")
    parser.add_argument("--test_data", type=str, required=True, help="Path to post-hoc extractions")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    test_path = Path(args.test_data)
    
    print(f"="*60)
    print(f"POST-HOC RATIONALIZATION TRANSFER TEST")
    print(f"="*60)
    print(f"Detector: {model_dir}")
    print(f"Test data: {test_path}")
    print()
    
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
    print(f"Loaded {len(examples)} examples")
    
    # Count by condition
    n_genuine = sum(1 for ex in examples if ex['condition'] == 'genuine')
    n_posthoc = sum(1 for ex in examples if ex['condition'] == 'posthoc')
    print(f"  Genuine (faithful): {n_genuine}")
    print(f"  Post-hoc (unfaithful): {n_posthoc}")
    
    # Get predictions
    probs, labels, conditions = predict(model, tokenizer, examples, args.device)
    
    # Overall evaluation
    results = evaluate(probs, labels)
    
    # Per-condition analysis
    genuine_mask = np.array([c == 'genuine' for c in conditions])
    posthoc_mask = np.array([c == 'posthoc' for c in conditions])
    
    genuine_probs = probs[genuine_mask]
    posthoc_probs = probs[posthoc_mask]
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS: Does hint-detector transfer to post-hoc detection?")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  AUROC:     {results['auroc']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP: {results['true_positives']:4d}  FP: {results['false_positives']:4d}")
    print(f"  FN: {results['false_negatives']:4d}  TN: {results['true_negatives']:4d}")
    
    print(f"\nProbability Distribution:")
    print(f"  Genuine (should be low):  mean={genuine_probs.mean():.3f}, std={genuine_probs.std():.3f}")
    print(f"  Post-hoc (should be high): mean={posthoc_probs.mean():.3f}, std={posthoc_probs.std():.3f}")
    print(f"  Separation: {posthoc_probs.mean() - genuine_probs.mean():.3f}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    if results['auroc'] >= 0.7:
        print("✓ TRANSFER DETECTED!")
        print(f"  The hint-detector (AUROC {results['auroc']:.3f}) can distinguish")
        print(f"  genuine reasoning from post-hoc rationalization.")
        print(f"  This suggests a common structural signature of unfaithful reasoning.")
    elif results['auroc'] >= 0.6:
        print("? WEAK TRANSFER")
        print(f"  Some signal detected (AUROC {results['auroc']:.3f}), but not strong.")
        print(f"  May need more data or different prompting strategy.")
    else:
        print("✗ NO TRANSFER")
        print(f"  Detector cannot distinguish (AUROC {results['auroc']:.3f}).")
        print(f"  Hint-detection and post-hoc detection may be different phenomena.")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = test_path.parent / f"posthoc_transfer_results_{test_path.stem}.json"
    
    output_data = {
        "model_dir": str(model_dir),
        "test_data": str(test_path),
        "n_examples": len(examples),
        "n_genuine": n_genuine,
        "n_posthoc": n_posthoc,
        "metrics": results,
        "genuine_prob_mean": float(genuine_probs.mean()),
        "posthoc_prob_mean": float(posthoc_probs.mean()),
        "separation": float(posthoc_probs.mean() - genuine_probs.mean()),
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
