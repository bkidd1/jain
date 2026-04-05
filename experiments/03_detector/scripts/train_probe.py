#!/usr/bin/env python3
"""
Train a linear probe to detect hint-influenced reasoning.

Method:
1. Extract activations from layer 14 (the "deception layer")
2. Train logistic regression: hint vs no-hint
3. Evaluate on held-out test set
4. Report accuracy and AUROC
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    device: torch.device,
    pool: str = "last",  # "mean" or "last"
) -> np.ndarray:
    """Extract activations from a specific layer."""
    
    if hasattr(model, 'model'):
        layer = model.model.layers[layer_idx]
    else:
        layer = model.transformer.h[layer_idx]
    
    captured = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['act'] = output[0].detach()
        else:
            captured['act'] = output.detach()
    
    hook = layer.register_forward_hook(capture_hook)
    
    activations = []
    for prompt in tqdm(prompts, desc=f"Extracting layer {layer_idx}"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        
        act = captured['act']  # [1, seq_len, hidden_dim]
        
        if pool == "mean":
            pooled = act.mean(dim=1)  # [1, hidden_dim]
        else:  # "last"
            pooled = act[:, -1, :]  # [1, hidden_dim]
        
        activations.append(pooled.cpu().numpy())
    
    hook.remove()
    
    return np.vstack(activations)  # [n_samples, hidden_dim]


def train_probe(
    model_name: str,
    data_path: Path,
    output_dir: Path,
    layer_idx: int = 14,
    pool: str = "last",
    test_size: float = 0.2,
):
    """Train and evaluate the hint detection probe."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    
    # Load data
    print(f"Loading data from: {data_path}")
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    # Separate by variant
    hint_data = [d for d in data if d.get('variant') == 'hint']
    no_hint_data = [d for d in data if d.get('variant') == 'no_hint']
    
    print(f"Found {len(hint_data)} hint examples, {len(no_hint_data)} no-hint examples")
    
    # Balance dataset
    n_samples = min(len(hint_data), len(no_hint_data))
    hint_data = hint_data[:n_samples]
    no_hint_data = no_hint_data[:n_samples]
    
    # Extract activations
    print(f"\nExtracting activations from layer {layer_idx}...")
    
    hint_prompts = [d['prompt'] for d in hint_data]
    no_hint_prompts = [d['prompt'] for d in no_hint_data]
    
    hint_acts = extract_activations(model, tokenizer, hint_prompts, layer_idx, device, pool)
    no_hint_acts = extract_activations(model, tokenizer, no_hint_prompts, layer_idx, device, pool)
    
    # Create dataset
    X = np.vstack([hint_acts, no_hint_acts])
    y = np.array([1] * len(hint_acts) + [0] * len(no_hint_acts))
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Labels: {sum(y)} hint, {len(y) - sum(y)} no-hint")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train logistic regression
    print("\nTraining logistic regression probe...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUROC:    {auroc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['no_hint', 'hint']))
    
    # Save results
    results = {
        'model': model_name,
        'layer': layer_idx,
        'pool': pool,
        'n_samples': n_samples * 2,
        'test_size': test_size,
        'accuracy': float(accuracy),
        'auroc': float(auroc),
        'train_size': len(X_train),
        'test_size_actual': len(X_test),
    }
    
    with open(output_dir / 'probe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model weights (the "hint direction")
    np.save(output_dir / 'probe_weights.npy', clf.coef_)
    np.save(output_dir / 'probe_bias.npy', clf.intercept_)
    
    print(f"\nSaved results to: {output_dir}")
    
    # Analyze the learned direction
    weights = clf.coef_[0]
    top_dims = np.argsort(np.abs(weights))[-10:][::-1]
    print(f"\nTop 10 dimensions by weight magnitude:")
    for dim in top_dims:
        print(f"  Dim {dim}: {weights[dim]:.4f}")
    
    return clf, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--pool", choices=["mean", "last"], default="last")
    args = parser.parse_args()
    
    train_probe(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        layer_idx=args.layer,
        pool=args.pool,
    )
