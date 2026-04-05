#!/usr/bin/env python3
"""
Activation Amplification: "Amplify the Brainwaves"

Inspired by Three-Body Problem's Trisolarans who can't lie because
their thoughts are visible. Can we amplify internal "hint-using"
signals so they surface in the model's output?

Method:
1. Extract activations from hint vs no-hint prompts
2. Find the "hint direction" = mean(hint) - mean(no_hint)
3. During generation, amplify activations along this direction
4. See if the model starts verbalizing hint usage
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
class AmplifyConfig:
    """Configuration for activation amplification."""
    amplify_factor: float = 2.0  # How much to amplify hint direction
    target_layers: tuple = None  # Which layers to amplify (None = middle layers)
    max_new_tokens: int = 256
    extract_layer: int = None  # Layer to extract direction from


class ActivationExtractor:
    """Extract activations from specific layers."""
    
    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = None
        self.hook = None
        self._setup_hook()
    
    def _setup_hook(self):
        """Register forward hook to capture activations."""
        if hasattr(self.model, 'model'):  # Llama-style
            layer = self.model.model.layers[self.layer_idx]
        elif hasattr(self.model, 'transformer'):  # GPT-style
            layer = self.model.transformer.h[self.layer_idx]
        else:
            raise ValueError("Unknown model architecture")
        
        self.hook = layer.register_forward_hook(self._capture_hook)
    
    def _capture_hook(self, module, input, output):
        """Capture the output activations."""
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def get_activations(self) -> torch.Tensor:
        """Get captured activations."""
        return self.activations
    
    def remove(self):
        """Remove the hook."""
        if self.hook:
            self.hook.remove()


class HintDirectionAmplifier:
    """
    Amplify the "hint direction" during generation.
    
    The hint direction is the difference between activations when
    the model sees a hint vs when it doesn't.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        hint_direction: torch.Tensor,
        amplify_factor: float,
        target_layers: list[int],
    ):
        self.model = model
        self.hint_direction = hint_direction
        self.amplify_factor = amplify_factor
        self.target_layers = target_layers
        self.hooks = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward hooks to amplify activations."""
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")
        
        for i in self.target_layers:
            hook = layers[i].register_forward_hook(self._amplify_hook)
            self.hooks.append(hook)
    
    def _amplify_hook(self, module, input, output):
        """Amplify activations along hint direction."""
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        # Project onto hint direction and amplify
        # hint_direction shape: [hidden_dim]
        # hidden shape: [batch, seq, hidden_dim]
        
        direction = self.hint_direction.to(hidden.device, hidden.dtype)
        direction = direction / (direction.norm() + 1e-8)  # Normalize
        
        # Compute projection: how much of each position is in hint direction
        projection = (hidden * direction).sum(dim=-1, keepdim=True)
        
        # Amplify: add more of the hint direction
        amplified = hidden + (self.amplify_factor - 1) * projection * direction
        
        if rest is not None:
            return (amplified,) + rest
        return amplified
    
    def remove(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def extract_hint_direction(
    model: nn.Module,
    tokenizer,
    hint_prompts: list[str],
    no_hint_prompts: list[str],
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract the "hint direction" from activation differences.
    
    Returns a vector pointing from no-hint to hint activations.
    """
    print(f"Extracting hint direction from layer {layer_idx}...")
    
    extractor = ActivationExtractor(model, layer_idx)
    
    hint_activations = []
    no_hint_activations = []
    
    # Extract hint activations
    for prompt in tqdm(hint_prompts, desc="Hint prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        # Take mean across sequence positions
        act = extractor.get_activations().mean(dim=1)  # [1, hidden]
        hint_activations.append(act)
    
    # Extract no-hint activations
    for prompt in tqdm(no_hint_prompts, desc="No-hint prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        act = extractor.get_activations().mean(dim=1)
        no_hint_activations.append(act)
    
    extractor.remove()
    
    # Compute direction: mean(hint) - mean(no_hint)
    hint_mean = torch.cat(hint_activations, dim=0).mean(dim=0)
    no_hint_mean = torch.cat(no_hint_activations, dim=0).mean(dim=0)
    
    direction = hint_mean - no_hint_mean
    
    print(f"Hint direction magnitude: {direction.norm():.4f}")
    
    return direction


def generate_with_amplification(
    model: nn.Module,
    tokenizer,
    prompt: str,
    amplifier: Optional[HintDirectionAmplifier],
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Generate text, optionally with hint amplification."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response


def run_experiment(
    model_name: str,
    data_path: Path,
    output_path: Path,
    config: Optional[AmplifyConfig] = None,
):
    """Run the amplification experiment."""
    if config is None:
        config = AmplifyConfig()
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    
    # Determine layers
    if hasattr(model, 'model'):
        num_layers = len(model.model.layers)
    else:
        num_layers = len(model.transformer.h)
    
    if config.extract_layer is None:
        config.extract_layer = num_layers // 2  # Middle layer
    
    if config.target_layers is None:
        # Amplify in middle-to-late layers
        start = num_layers // 2
        end = int(num_layers * 0.9)
        config.target_layers = list(range(start, end))
    
    print(f"Extract layer: {config.extract_layer}")
    print(f"Amplify layers: {config.target_layers}")
    
    # Load data
    print(f"Loading data from: {data_path}")
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    # Separate hint and no-hint prompts
    hint_prompts = [d['prompt'] for d in data if d.get('variant') == 'hint']
    no_hint_prompts = [d['prompt'] for d in data if d.get('variant') == 'no_hint']
    
    print(f"Found {len(hint_prompts)} hint prompts, {len(no_hint_prompts)} no-hint prompts")
    
    # If we only have hint variants, create synthetic no-hint versions
    if len(no_hint_prompts) == 0:
        print("No no-hint variants found, using hint prompts only for direction")
        # Use a simple baseline: remove the hint context line
        no_hint_prompts = []
        for p in hint_prompts[:10]:
            # Try to remove hint-like phrases
            lines = p.split('\n')
            filtered = [l for l in lines if 'believes' not in l.lower() and 'thinks' not in l.lower()]
            no_hint_prompts.append('\n'.join(filtered))
    
    # Extract hint direction (use subset for speed)
    hint_direction = extract_hint_direction(
        model, tokenizer,
        hint_prompts[:20],
        no_hint_prompts[:20],
        config.extract_layer,
        device,
    )
    
    # Set up amplifier
    amplifier = HintDirectionAmplifier(
        model,
        hint_direction,
        config.amplify_factor,
        config.target_layers,
    )
    
    # Run comparisons
    results = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Test on hint prompts - compare normal vs amplified
    test_prompts = hint_prompts[:10]
    
    with open(output_path, 'w') as f:
        for i, prompt in enumerate(tqdm(test_prompts, desc="Testing")):
            # Generate WITHOUT amplification
            amplifier.remove()
            normal_response = generate_with_amplification(
                model, tokenizer, prompt, None,
                config.max_new_tokens, device
            )
            
            # Re-setup amplifier
            amplifier = HintDirectionAmplifier(
                model, hint_direction,
                config.amplify_factor, config.target_layers
            )
            
            # Generate WITH amplification
            amplified_response = generate_with_amplification(
                model, tokenizer, prompt, amplifier,
                config.max_new_tokens, device
            )
            
            result = {
                'prompt': prompt,
                'normal_response': normal_response,
                'amplified_response': amplified_response,
                'amplify_factor': config.amplify_factor,
            }
            results.append(result)
            
            f.write(json.dumps(result) + '\n')
            f.flush()
            
            # Print comparison
            print(f"\n{'='*60}")
            print(f"Example {i+1}")
            print(f"{'='*60}")
            print(f"NORMAL:\n{normal_response[:500]}...")
            print(f"\nAMPLIFIED (factor={config.amplify_factor}):\n{amplified_response[:500]}...")
    
    amplifier.remove()
    print(f"\nSaved results to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amplify", type=float, default=2.0)
    args = parser.parse_args()
    
    config = AmplifyConfig(amplify_factor=args.amplify)
    
    run_experiment(
        model_name=args.model,
        data_path=args.data,
        output_path=args.output,
        config=config,
    )
