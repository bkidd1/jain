#!/usr/bin/env python3
"""
Test probe training on Gemma 4 to see if hint detection generalizes.

Key questions:
1. Does Gemma 4 also encode hint-influence in a linearly separable way?
2. Which layer(s) show the best separation?
3. Is there a similar "two-stage" processing pattern?
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse


def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    device: torch.device,
    pool: str = "last",
) -> np.ndarray:
    """Extract activations from a specific layer."""
    
    # Handle different model architectures
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        # Gemma 4 multimodal architecture
        layer = model.model.language_model.layers[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layer = model.transformer.h[layer_idx]
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")
    
    captured = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['act'] = output[0].detach()
        else:
            captured['act'] = output.detach()
    
    hook = layer.register_forward_hook(capture_hook)
    
    activations = []
    for prompt in tqdm(prompts, desc=f"Layer {layer_idx}", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        
        act = captured['act']
        
        if pool == "mean":
            pooled = act.mean(dim=1)
        else:
            pooled = act[:, -1, :]
        
        activations.append(pooled.cpu().numpy())
    
    hook.remove()
    
    return np.vstack(activations)


def train_and_eval_probe(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """Train logistic regression probe and return metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    
    return accuracy, auroc, clf


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load with appropriate quantization for memory constraints
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    device = next(model.parameters()).device
    
    # Get model info
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        # Gemma 4 multimodal architecture
        n_layers = len(model.model.language_model.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        n_layers = len(model.transformer.h)
    else:
        n_layers = args.n_layers or 32
    
    print(f"Model has {n_layers} layers")
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    
    # Separate by variant
    hint_data = [d for d in data if d.get('variant') == 'hint']
    no_hint_data = [d for d in data if d.get('variant') == 'no_hint']
    
    # Balance and limit
    n_samples = min(len(hint_data), len(no_hint_data), args.max_samples)
    hint_data = hint_data[:n_samples]
    no_hint_data = no_hint_data[:n_samples]
    
    print(f"Using {n_samples} samples per class")
    
    hint_prompts = [d['prompt'] for d in hint_data]
    no_hint_prompts = [d['prompt'] for d in no_hint_data]
    
    # Determine layers to test
    if args.layers:
        test_layers = [int(l) for l in args.layers.split(',')]
    else:
        # Test a spread of layers
        test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        # Also test every layer if doing full sweep
        if args.full_sweep:
            test_layers = list(range(n_layers))
    
    print(f"\nTesting layers: {test_layers}")
    
    results = []
    
    for layer_idx in tqdm(test_layers, desc="Layer sweep"):
        if layer_idx >= n_layers:
            print(f"Skipping layer {layer_idx} (model only has {n_layers} layers)")
            continue
            
        try:
            hint_acts = extract_activations(model, tokenizer, hint_prompts, layer_idx, device)
            no_hint_acts = extract_activations(model, tokenizer, no_hint_prompts, layer_idx, device)
            
            X = np.vstack([hint_acts, no_hint_acts])
            y = np.array([1] * len(hint_acts) + [0] * len(no_hint_acts))
            
            accuracy, auroc, clf = train_and_eval_probe(X, y)
            
            result = {
                'layer': layer_idx,
                'accuracy': float(accuracy),
                'auroc': float(auroc),
                'hidden_dim': hint_acts.shape[1],
            }
            results.append(result)
            
            print(f"  Layer {layer_idx}: acc={accuracy:.3f}, auroc={auroc:.3f}")
            
        except Exception as e:
            print(f"  Layer {layer_idx}: ERROR - {e}")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Samples: {n_samples * 2} ({n_samples} per class)")
    print()
    
    if results:
        best = max(results, key=lambda x: x['auroc'])
        print(f"Best layer: {best['layer']}")
        print(f"  Accuracy: {best['accuracy']:.3f}")
        print(f"  AUROC:    {best['auroc']:.3f}")
        
        # Check for two-stage pattern
        early_layers = [r for r in results if r['layer'] < n_layers // 3]
        late_layers = [r for r in results if r['layer'] >= 2 * n_layers // 3]
        
        if early_layers and late_layers:
            early_best = max(early_layers, key=lambda x: x['auroc'])
            late_best = max(late_layers, key=lambda x: x['auroc'])
            
            print(f"\nTwo-stage analysis:")
            print(f"  Early best (L{early_best['layer']}): {early_best['auroc']:.3f}")
            print(f"  Late best (L{late_best['layer']}):  {late_best['auroc']:.3f}")
            
            if late_best['auroc'] > early_best['auroc'] + 0.05:
                print("  → Late layers show better separation (consistent with two-stage hypothesis)")
    
    # Save results
    output = {
        'model': args.model,
        'n_samples': n_samples * 2,
        'layers_tested': test_layers,
        'results': results,
    }
    
    with open(output_dir / 'gemma4_probe_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved results to: {output_dir / 'gemma4_probe_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B")
    parser.add_argument("--data", type=Path, default=Path("~/jain/experiments/03_detector/data/combined_v2.jsonl").expanduser())
    parser.add_argument("--output", type=Path, default=Path("~/jain/experiments/05_cross_model/results").expanduser())
    parser.add_argument("--layers", help="Comma-separated layer indices to test")
    parser.add_argument("--full-sweep", action="store_true", help="Test all layers")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per class")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--n-layers", type=int, help="Number of layers if auto-detect fails")
    args = parser.parse_args()
    
    main(args)
