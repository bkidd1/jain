#!/usr/bin/env python3
"""
Noise Injection for Unfaithfulness Detection

Based on Liu et al. (ICLR 2026): "Enhancing Hallucination Detection through Noise Injection"
https://arxiv.org/abs/2502.03799

Core idea: Inject uniform noise into MLP activations during inference.
Hypothesis: Unfaithful (hint-influenced) CoT is more fragile under noise than faithful CoT.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    alpha: float = 0.07  # Noise magnitude (from Liu et al.)
    layer_start_frac: float = 0.6  # Start injecting at 60% of layers
    layer_end_frac: float = 1.0  # End at final layer
    temperature: float = 0.5  # Sampling temperature
    num_samples: int = 10  # K samples per question
    max_new_tokens: int = 256


class MLPNoiseInjector:
    """
    Hook-based noise injection into MLP activations.
    
    Following Liu et al.: 
    - Inject same noise vector across all target layers (skip connections cancel independent noise)
    - Uniform(0, alpha) noise added to MLP outputs
    """
    
    def __init__(self, model: nn.Module, config: NoiseConfig):
        self.model = model
        self.config = config
        self.hooks = []
        self.noise_vector = None
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward hooks on MLP layers."""
        # Get total number of layers
        if hasattr(self.model, 'model'):  # Llama-style
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):  # GPT-style
            layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")
        
        num_layers = len(layers)
        start_layer = int(num_layers * self.config.layer_start_frac)
        end_layer = int(num_layers * self.config.layer_end_frac)
        
        print(f"Injecting noise into layers {start_layer}-{end_layer} of {num_layers}")
        
        for i, layer in enumerate(layers):
            if start_layer <= i < end_layer:
                # Hook into MLP output
                if hasattr(layer, 'mlp'):  # Llama-style
                    hook = layer.mlp.register_forward_hook(self._noise_hook)
                elif hasattr(layer, 'mlp'):  # GPT-style  
                    hook = layer.mlp.register_forward_hook(self._noise_hook)
                else:
                    continue
                self.hooks.append(hook)
    
    def _noise_hook(self, module, input, output):
        """Add noise to MLP output."""
        if self.noise_vector is None:
            return output
        
        # Ensure noise vector matches output shape
        if self.noise_vector.shape[-1] != output.shape[-1]:
            # Regenerate noise for correct hidden dim
            self.noise_vector = torch.empty(
                output.shape[-1], 
                device=output.device,
                dtype=output.dtype
            ).uniform_(0, self.config.alpha)
        
        # Add noise (broadcast across batch and sequence)
        return output + self.noise_vector
    
    def sample_noise(self, hidden_dim: int, device: torch.device, dtype: torch.dtype):
        """Sample a new noise vector (reused across layers per Liu et al.)"""
        self.noise_vector = torch.empty(
            hidden_dim, device=device, dtype=dtype
        ).uniform_(0, self.config.alpha)
    
    def clear_noise(self):
        """Clear noise vector (for clean generation)."""
        self.noise_vector = None
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def generate_with_noise(
    model: nn.Module,
    tokenizer,
    prompt: str,
    config: NoiseConfig,
    injector: MLPNoiseInjector,
) -> list[str]:
    """
    Generate K samples with noise injection.
    
    Returns list of generated responses.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Get hidden dim from model config
    hidden_dim = model.config.hidden_size
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    responses = []
    for _ in range(config.num_samples):
        # Sample new noise vector for this generation
        injector.sample_noise(hidden_dim, device, dtype)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode (skip prompt tokens)
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        responses.append(response)
    
    return responses


def extract_final_answer(response: str) -> Optional[str]:
    """
    Extract final answer from CoT response.
    
    Looks for patterns like:
    - "The answer is X"
    - "#### X"
    - Final number in response
    """
    import re
    
    # Pattern 1: "The answer is X"
    match = re.search(r'[Tt]he answer is[:\s]*([^\n.]+)', response)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: "#### X" (GSM8K style)
    match = re.search(r'####\s*([^\n]+)', response)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: Last number in response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]
    
    return None


def compute_answer_entropy(responses: list[str]) -> float:
    """
    Compute answer entropy over K responses.
    
    H = -Σ p(a) log p(a)
    """
    answers = [extract_final_answer(r) for r in responses]
    answers = [a for a in answers if a is not None]
    
    if not answers:
        return float('nan')
    
    # Count occurrences
    from collections import Counter
    counts = Counter(answers)
    total = len(answers)
    
    # Compute entropy
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)
    
    return entropy


def run_experiment(
    model_name: str,
    data_path: Path,
    output_path: Path,
    config: Optional[NoiseConfig] = None,
    hint_only: bool = True,
):
    """
    Run noise injection experiment on dataset.
    
    Args:
        model_name: HuggingFace model name
        data_path: Path to JSONL with v1 extraction format
        output_path: Where to save results
        config: NoiseConfig settings
        hint_only: If True, only process hint-variant examples (where unfaithful can occur)
    """
    if config is None:
        config = NoiseConfig()
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    # Setup noise injector
    injector = MLPNoiseInjector(model, config)
    
    # Load data
    print(f"Loading data from: {data_path}")
    data = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            # Filter to hint-variant only if specified (unfaithful only occurs with hints)
            if hint_only and item.get('variant') != 'hint':
                continue
            data.append(item)
    
    print(f"Loaded {len(data)} examples (hint_only={hint_only})")
    
    # Count labels
    n_unfaithful = sum(1 for d in data if d.get('label') == 'unfaithful')
    n_faithful = len(data) - n_unfaithful
    print(f"  Unfaithful: {n_unfaithful}, Faithful: {n_faithful}")
    
    results = []
    
    # Save incrementally to avoid losing progress
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in tqdm(data, desc="Processing"):
            prompt = item['prompt']
            
            # Generate K samples with noise
            responses = generate_with_noise(
                model, tokenizer, prompt, config, injector
            )
            
            # Compute metrics
            entropy = compute_answer_entropy(responses)
            
            # Use label from v1 data: "unfaithful" = model used the hint
            is_unfaithful = item.get('label') == 'unfaithful'
            
            result = {
                'pair_id': item.get('pair_id'),
                'variant': item.get('variant'),
                'prompt': prompt,
                'original_response': item.get('response'),
                'correct_answer': item.get('correct_answer'),
                'misleading_answer': item.get('misleading_answer'),
                'label': item.get('label'),
                'is_unfaithful': is_unfaithful,
                'noise_responses': responses,
                'answer_entropy': entropy,
            }
            results.append(result)
            
            # Write incrementally
            f.write(json.dumps(result) + '\n')
            f.flush()
    
    # Cleanup
    injector.remove_hooks()
    
    print(f"Saved results to: {output_path}")
    
    # Quick summary
    unfaithful_entropies = [r['answer_entropy'] for r in results if r['is_unfaithful'] and not np.isnan(r['answer_entropy'])]
    faithful_entropies = [r['answer_entropy'] for r in results if not r['is_unfaithful'] and not np.isnan(r['answer_entropy'])]
    
    print(f"\n=== Quick Summary ===")
    print(f"Unfaithful entropy:  {np.mean(unfaithful_entropies):.3f} ± {np.std(unfaithful_entropies):.3f} (n={len(unfaithful_entropies)})")
    print(f"Faithful entropy:    {np.mean(faithful_entropies):.3f} ± {np.std(faithful_entropies):.3f} (n={len(faithful_entropies)})")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()
    
    config = NoiseConfig(
        alpha=args.alpha,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )
    
    run_experiment(
        model_name=args.model,
        data_path=args.data,
        output_path=args.output,
        config=config,
    )
