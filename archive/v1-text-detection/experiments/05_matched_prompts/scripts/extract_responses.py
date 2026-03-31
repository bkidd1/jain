#!/usr/bin/env python3
"""
Extract CoT responses from models using matched prompts.
Uses logit lens to label whether hint was processed internally.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_top_tokens_at_position(model, tokenizer, input_ids, position: int, k: int = 5):
    """Get top-k predicted tokens at a specific position using logit lens."""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
        
        top_tokens_by_layer = []
        for layer_idx, hidden in enumerate(hidden_states[1:]):  # skip embedding layer
            # Project to vocabulary
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(hidden[0, position, :])
            else:
                logits = model.model.embed_tokens.weight @ hidden[0, position, :]
            
            # Get top-k
            top_k = torch.topk(logits, k)
            tokens = [tokenizer.decode([idx]) for idx in top_k.indices.tolist()]
            top_tokens_by_layer.append(tokens)
        
        return top_tokens_by_layer


def check_hint_in_trace(top_tokens_by_layer: list, hint_answer: str) -> tuple[bool, list]:
    """Check if hint answer appears in any layer's top tokens."""
    hint_lower = hint_answer.lower()
    hint_parts = hint_lower.split()
    
    appearances = []
    for layer_idx, tokens in enumerate(top_tokens_by_layer):
        tokens_lower = [t.lower().strip() for t in tokens]
        for part in hint_parts:
            if any(part in t for t in tokens_lower):
                appearances.append(layer_idx)
                break
    
    return len(appearances) > 0, appearances


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate a CoT response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_from_model(model_name: str, pairs_path: Path, output_path: Path, device: str = "mps"):
    """Extract responses and labels from a single model."""
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    
    # Load pairs
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    
    results = []
    for pair in tqdm(pairs, desc=f"Extracting from {model_name.split('/')[-1]}"):
        # Process both conditions
        for variant in ["no_hint", "hint"]:
            prompt_key = f"{variant}_prompt"
            prompt = pair[prompt_key]
            
            # Generate response
            response = generate_response(model, tokenizer, prompt)
            
            # Get logit lens trace at last input position
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            last_pos = inputs.input_ids.shape[1] - 1
            top_tokens = get_top_tokens_at_position(model, tokenizer, inputs.input_ids, last_pos)
            
            # Check for hint in trace (only meaningful for hint condition)
            hint_answer = pair["misleading_answer"]
            hint_in_trace, appearances = check_hint_in_trace(top_tokens, hint_answer)
            
            # Determine label
            if variant == "no_hint":
                label = "faithful"  # No hint to be influenced by
            else:
                # Hint condition: unfaithful if hint appears in internal trace
                label = "unfaithful" if hint_in_trace else "faithful"
            
            result = {
                "pair_id": pair["id"],
                "variant": variant,
                "prompt": prompt,
                "response": response,
                "correct_answer": pair["correct_answer"],
                "misleading_answer": hint_answer,
                "hint_in_trace": hint_in_trace,
                "hint_layer_appearances": appearances,
                "label": label,
                "source_model": model_name.split("/")[-1],
            }
            results.append(result)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Stats
    n_unfaithful = sum(1 for r in results if r["label"] == "unfaithful")
    print(f"Saved {len(results)} examples ({n_unfaithful} unfaithful)")
    print(f"Output: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pairs", type=str, default="data/hint_pairs/matched_pairs.jsonl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    pairs_path = script_dir / args.pairs
    
    if args.output:
        output_path = script_dir / args.output
    else:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        output_path = script_dir / f"data/extractions/{model_short}_matched.jsonl"
    
    extract_from_model(args.model, pairs_path, output_path, args.device)


if __name__ == "__main__":
    main()
