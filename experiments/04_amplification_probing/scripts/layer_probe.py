#!/usr/bin/env python3
"""
Amplification Probing: Systematic layer-by-layer analysis with quantifiable metrics.

For each layer:
1. Extract direction (hint vs no-hint)
2. Generate with/without amplification
3. Measure internal changes (logits, entropy, alignment)
4. Measure external changes (text output)
5. Build semantic map of what each layer encodes
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Optional


@dataclass
class LayerMetrics:
    """Quantifiable metrics for a single layer."""
    layer: int
    direction_magnitude: float
    
    # Internal metrics (averaged over test prompts)
    kl_divergence: float          # KL(amplified || baseline) on logits
    entropy_baseline: float       # Entropy of baseline distribution
    entropy_amplified: float      # Entropy of amplified distribution
    entropy_change: float         # Relative change
    
    # Token probability shifts
    top_tokens_changed: int       # How many of top-10 tokens changed
    max_prob_shift: float         # Largest probability change for any token
    
    # Direction alignment
    activation_alignment: float   # Cosine sim of activations with direction
    
    # Output changes
    output_changed: bool          # Did the text output change?
    baseline_response: str
    amplified_response: str


def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P || Q) where P=amplified, Q=baseline."""
    # Use log_softmax for numerical stability
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    # KL = sum(p * (log_p - log_q))
    kl = (p * (log_p - log_q)).sum(dim=-1)
    # Clamp to valid range (numerical errors can give negative values)
    kl = kl.clamp(min=0.0)
    result = kl.mean().item()
    # Handle potential NaN from empty or degenerate cases
    return result if not np.isnan(result) else 0.0


def compute_entropy(logits: torch.Tensor) -> float:
    """Compute entropy of distribution."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    # H = -sum(p * log_p)
    entropy = -(p * log_p).sum(dim=-1)
    result = entropy.mean().item()
    return result if not np.isnan(result) else 0.0


def count_top_k_changes(logits_base: torch.Tensor, logits_amp: torch.Tensor, k: int = 10) -> int:
    """Count how many of top-k tokens differ."""
    top_base = torch.topk(logits_base, k, dim=-1).indices
    top_amp = torch.topk(logits_amp, k, dim=-1).indices
    # Check overlap
    changed = 0
    for i in range(logits_base.shape[0]):
        base_set = set(top_base[i].tolist())
        amp_set = set(top_amp[i].tolist())
        changed += k - len(base_set & amp_set)
    return changed // logits_base.shape[0]  # Average


def max_probability_shift(logits_base: torch.Tensor, logits_amp: torch.Tensor) -> float:
    """Maximum probability shift for any token."""
    p_base = F.softmax(logits_base, dim=-1)
    p_amp = F.softmax(logits_amp, dim=-1)
    max_shift = (p_amp - p_base).abs().max().item()
    return max_shift


def extract_direction(model, tokenizer, hint_prompts, no_hint_prompts, layer_idx, device):
    """Extract direction from a single layer."""
    if hasattr(model, 'model'):
        layer = model.model.layers[layer_idx]
    else:
        layer = model.transformer.h[layer_idx]
    
    captured = {}
    def capture(m, i, o):
        captured['act'] = o[0].detach() if isinstance(o, tuple) else o.detach()
    
    hook = layer.register_forward_hook(capture)
    
    hint_acts = []
    for p in hint_prompts:
        inputs = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inputs)
        hint_acts.append(captured['act'].mean(dim=1))
    
    no_hint_acts = []
    for p in no_hint_prompts:
        inputs = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inputs)
        no_hint_acts.append(captured['act'].mean(dim=1))
    
    hook.remove()
    
    hint_mean = torch.cat(hint_acts, dim=0).mean(dim=0)
    no_hint_mean = torch.cat(no_hint_acts, dim=0).mean(dim=0)
    direction = hint_mean - no_hint_mean
    
    return direction


def probe_layer(
    model,
    tokenizer,
    direction: torch.Tensor,
    layer_idx: int,
    test_prompts: list[str],
    factor: float,
    device: torch.device,
) -> LayerMetrics:
    """Probe a single layer with amplification and collect metrics."""
    
    if hasattr(model, 'model'):
        layer = model.model.layers[layer_idx]
    else:
        layer = model.transformer.h[layer_idx]
    
    direction_norm = direction / (direction.norm() + 1e-8)
    
    # Collect metrics across test prompts
    kl_divs = []
    entropy_bases = []
    entropy_amps = []
    top_k_changes = []
    prob_shifts = []
    alignments = []
    
    baseline_responses = []
    amplified_responses = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        # --- Baseline generation ---
        with torch.no_grad():
            outputs_base = model(**inputs)
            logits_base = outputs_base.logits[:, -1, :]  # Last token logits
        
        # Generate baseline text
        with torch.no_grad():
            out_base = model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        resp_base = tokenizer.decode(out_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)[:100]
        baseline_responses.append(resp_base)
        
        # --- Amplified generation ---
        captured_act = {}
        
        def amplify_hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            rest = o[1:] if isinstance(o, tuple) else None
            
            d = direction_norm.to(h.device, h.dtype)
            proj = (h * d).sum(dim=-1, keepdim=True)
            captured_act['alignment'] = (proj.abs() / (h.norm(dim=-1, keepdim=True) + 1e-8)).mean().item()
            
            amp = h + (factor - 1) * proj * d
            return (amp,) + rest if rest else amp
        
        hook = layer.register_forward_hook(amplify_hook)
        
        with torch.no_grad():
            outputs_amp = model(**inputs)
            logits_amp = outputs_amp.logits[:, -1, :]
        
        # Generate amplified text
        with torch.no_grad():
            out_amp = model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        resp_amp = tokenizer.decode(out_amp[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)[:100]
        amplified_responses.append(resp_amp)
        
        hook.remove()
        
        # Compute metrics
        kl_divs.append(compute_kl_divergence(logits_amp, logits_base))
        entropy_bases.append(compute_entropy(logits_base))
        entropy_amps.append(compute_entropy(logits_amp))
        top_k_changes.append(count_top_k_changes(logits_base, logits_amp))
        prob_shifts.append(max_probability_shift(logits_base, logits_amp))
        alignments.append(captured_act.get('alignment', 0))
    
    # Aggregate metrics
    entropy_base_mean = np.mean(entropy_bases)
    entropy_amp_mean = np.mean(entropy_amps)
    
    return LayerMetrics(
        layer=layer_idx,
        direction_magnitude=direction.norm().item(),
        kl_divergence=np.mean(kl_divs),
        entropy_baseline=entropy_base_mean,
        entropy_amplified=entropy_amp_mean,
        entropy_change=(entropy_amp_mean - entropy_base_mean) / (entropy_base_mean + 1e-8),
        top_tokens_changed=int(np.mean(top_k_changes)),
        max_prob_shift=np.mean(prob_shifts),
        activation_alignment=np.mean(alignments),
        output_changed=any(b != a for b, a in zip(baseline_responses, amplified_responses)),
        baseline_response=baseline_responses[0] if baseline_responses else "",
        amplified_response=amplified_responses[0] if amplified_responses else "",
    )


def run_full_probe(
    model_name: str,
    data_path: Path,
    output_path: Path,
    factor: float = 1.2,
    n_direction_samples: int = 20,
    n_test_samples: int = 5,
):
    """Run full layer-by-layer probing."""
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    
    # Get number of layers
    if hasattr(model, 'model'):
        num_layers = len(model.model.layers)
    else:
        num_layers = len(model.transformer.h)
    
    print(f"Model has {num_layers} layers")
    
    # Load data
    data = [json.loads(l) for l in open(data_path)]
    hint_prompts = [d['prompt'] for d in data if d.get('variant') == 'hint'][:n_direction_samples]
    no_hint_prompts = [d['prompt'] for d in data if d.get('variant') == 'no_hint'][:n_direction_samples]
    test_prompts = hint_prompts[:n_test_samples]
    
    print(f"Using {len(hint_prompts)} hint, {len(no_hint_prompts)} no-hint for direction")
    print(f"Testing on {len(test_prompts)} prompts")
    
    results = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for layer_idx in tqdm(range(num_layers), desc="Probing layers"):
            # Extract direction
            direction = extract_direction(
                model, tokenizer, hint_prompts, no_hint_prompts, layer_idx, device
            )
            
            # Probe layer
            metrics = probe_layer(
                model, tokenizer, direction, layer_idx, test_prompts, factor, device
            )
            
            results.append(metrics)
            f.write(json.dumps(asdict(metrics)) + '\n')
            f.flush()
            
            print(f"Layer {layer_idx:2d}: KL={metrics.kl_divergence:.4f}, "
                  f"ΔH={metrics.entropy_change:+.3f}, "
                  f"top-k Δ={metrics.top_tokens_changed}, "
                  f"output_changed={metrics.output_changed}")
    
    # Summary
    print("\n" + "="*70)
    print("LAYER-BY-LAYER SUMMARY")
    print("="*70)
    print(f"{'Layer':<6} {'KL Div':<10} {'Entropy Δ':<12} {'Top-k Δ':<10} {'Output Δ':<10}")
    print("-"*50)
    for m in results:
        print(f"{m.layer:<6} {m.kl_divergence:<10.4f} {m.entropy_change:<+12.3f} "
              f"{m.top_tokens_changed:<10} {str(m.output_changed):<10}")
    
    # Find most impactful layers
    by_kl = sorted(results, key=lambda x: x.kl_divergence, reverse=True)[:3]
    print(f"\nTop 3 layers by KL divergence: {[m.layer for m in by_kl]}")
    
    by_output = [m for m in results if m.output_changed]
    print(f"Layers that changed output: {[m.layer for m in by_output]}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=float, default=1.2)
    args = parser.parse_args()
    
    run_full_probe(
        model_name=args.model,
        data_path=args.data,
        output_path=args.output,
        factor=args.factor,
    )
