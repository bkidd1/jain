#!/usr/bin/env python3
"""
Temperature Sweep for Unfaithfulness Detection

Hypothesis: Unfaithful responses may show different sensitivity to temperature
than faithful responses.

Method: Generate responses at multiple temperatures, measure answer variance.
"""

import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from collections import Counter


@dataclass
class TempConfig:
    """Configuration for temperature sweep."""
    temperatures: tuple = (0.1, 0.3, 0.5, 0.7, 1.0)
    samples_per_temp: int = 3  # K samples at each temperature
    max_new_tokens: int = 256


def extract_final_answer(response: str) -> Optional[str]:
    """Extract final answer from response."""
    import re
    
    # Pattern 1: "The answer is X"
    match = re.search(r'[Tt]he answer is[:\s]*([^\n.]+)', response)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: "#### X" (GSM8K style)
    match = re.search(r'####\s*([^\n]+)', response)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: Last number
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]
    
    return None


def generate_at_temperature(
    model, tokenizer, prompt: str, temp: float, 
    num_samples: int, max_new_tokens: int, device
) -> list[str]:
    """Generate multiple samples at a specific temperature."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    responses = []
    for _ in range(num_samples):
        with torch.no_grad():
            if temp < 0.01:  # Effectively greedy
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        responses.append(response)
    
    return responses


def compute_answer_entropy(responses: list[str]) -> float:
    """Compute entropy over answers."""
    answers = [extract_final_answer(r) for r in responses]
    answers = [a for a in answers if a is not None]
    
    if not answers:
        return float('nan')
    
    counts = Counter(answers)
    total = len(answers)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)
    
    return entropy


def compute_temperature_sensitivity(temp_responses: dict) -> dict:
    """
    Compute metrics across temperature sweep.
    
    Returns:
        - entropy_by_temp: entropy at each temperature
        - entropy_slope: how entropy changes with temperature
        - total_unique_answers: unique answers across all temps
        - answer_stability: fraction of temps giving same majority answer
    """
    entropies = {}
    all_answers = []
    majority_answers = []
    
    for temp, responses in sorted(temp_responses.items()):
        entropy = compute_answer_entropy(responses)
        entropies[temp] = entropy
        
        answers = [extract_final_answer(r) for r in responses]
        answers = [a for a in answers if a is not None]
        all_answers.extend(answers)
        
        if answers:
            majority = Counter(answers).most_common(1)[0][0]
            majority_answers.append(majority)
    
    # Compute entropy slope (linear regression)
    temps = sorted(temp_responses.keys())
    ent_values = [entropies[t] for t in temps if not np.isnan(entropies[t])]
    
    if len(ent_values) >= 2:
        # Simple slope: (last - first) / (temp_range)
        entropy_slope = (ent_values[-1] - ent_values[0]) / (temps[-1] - temps[0])
    else:
        entropy_slope = float('nan')
    
    # Answer stability: how often majority answer is the same
    if majority_answers:
        most_common_majority = Counter(majority_answers).most_common(1)[0][1]
        answer_stability = most_common_majority / len(majority_answers)
    else:
        answer_stability = float('nan')
    
    return {
        'entropy_by_temp': entropies,
        'entropy_slope': entropy_slope,
        'total_unique_answers': len(set(all_answers)),
        'answer_stability': answer_stability,
        'mean_entropy': np.nanmean(list(entropies.values())),
    }


def run_experiment(
    model_name: str,
    data_path: Path,
    output_path: Path,
    config: Optional[TempConfig] = None,
):
    """Run temperature sweep experiment."""
    if config is None:
        config = TempConfig()
    
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
            item = json.loads(line)
            if item.get('variant') == 'hint':
                data.append(item)
    
    print(f"Loaded {len(data)} examples")
    n_unfaithful = sum(1 for d in data if d.get('label') == 'unfaithful')
    print(f"  Unfaithful: {n_unfaithful}, Faithful: {len(data) - n_unfaithful}")
    
    # Save incrementally
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    
    with open(output_path, 'w') as f:
        for item in tqdm(data, desc="Processing"):
            prompt = item['prompt']
            
            # Generate at each temperature
            temp_responses = {}
            for temp in config.temperatures:
                responses = generate_at_temperature(
                    model, tokenizer, prompt, temp,
                    config.samples_per_temp, config.max_new_tokens, device
                )
                temp_responses[temp] = responses
            
            # Compute sensitivity metrics
            sensitivity = compute_temperature_sensitivity(temp_responses)
            
            result = {
                'pair_id': item.get('pair_id'),
                'label': item.get('label'),
                'is_unfaithful': item.get('label') == 'unfaithful',
                'temp_responses': {str(k): v for k, v in temp_responses.items()},
                **sensitivity,
            }
            results.append(result)
            
            f.write(json.dumps(result) + '\n')
            f.flush()
    
    print(f"Saved results to: {output_path}")
    
    # Quick summary
    unfaithful = [r for r in results if r['is_unfaithful']]
    faithful = [r for r in results if not r['is_unfaithful']]
    
    print(f"\n=== Quick Summary ===")
    print(f"Unfaithful mean_entropy:  {np.nanmean([r['mean_entropy'] for r in unfaithful]):.3f}")
    print(f"Faithful mean_entropy:    {np.nanmean([r['mean_entropy'] for r in faithful]):.3f}")
    print(f"Unfaithful entropy_slope: {np.nanmean([r['entropy_slope'] for r in unfaithful]):.3f}")
    print(f"Faithful entropy_slope:   {np.nanmean([r['entropy_slope'] for r in faithful]):.3f}")
    print(f"Unfaithful stability:     {np.nanmean([r['answer_stability'] for r in unfaithful]):.3f}")
    print(f"Faithful stability:       {np.nanmean([r['answer_stability'] for r in faithful]):.3f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-temp", type=int, default=3)
    args = parser.parse_args()
    
    config = TempConfig(samples_per_temp=args.samples_per_temp)
    
    run_experiment(
        model_name=args.model,
        data_path=args.data,
        output_path=args.output,
        config=config,
    )
