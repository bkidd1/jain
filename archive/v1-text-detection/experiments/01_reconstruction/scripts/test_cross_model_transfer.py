#!/usr/bin/env python3
"""
Week 5-6: The Money Shot — Cross-Model Transfer Test

Can an RTP trained on Llama's internal traces predict Mistral's reasoning?

This is the core experiment. If it works, we have a publishable result.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer


def load_rtp(model_path: str = "models/rtp_v1"):
    """Load the trained RTP model."""
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer, device


def get_mistral_trace(model, prompt: str) -> dict:
    """Extract actual trace from Mistral using logit lens."""
    
    # Check 75% layer (where reasoning crystallizes)
    n_layers = model.cfg.n_layers
    target_layer = 3 * n_layers // 4
    
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        
        resid = cache[f"blocks.{target_layer}.hook_resid_post"]
        normalized = model.ln_final(resid)
        logits = model.unembed(normalized)
        probs = torch.softmax(logits[0, -1], dim=-1)
        
        top = torch.topk(probs, 5)
        concepts = [
            (model.tokenizer.decode([idx]).strip(), prob.item())
            for prob, idx in zip(top.values, top.indices)
        ]
        
        # Get final output
        final_logits = model(prompt)
        final_probs = torch.softmax(final_logits[0, -1], dim=-1)
        final_token = model.tokenizer.decode([final_probs.argmax().item()]).strip()
    
    return {
        "final_output": final_token,
        "internal_concepts": concepts,
        "top_concept": concepts[0][0] if concepts else ""
    }


def predict_trace(rtp_model, tokenizer, prompt: str, answer: str, device: str) -> str:
    """Use RTP to predict the reasoning trace."""
    
    input_text = f"Prompt: {prompt}\nAnswer: {answer}\nTrace:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = rtp_model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    trace = generated.split("Trace:")[-1].strip().split("\n")[0]
    
    return trace


def evaluate_overlap(predicted: str, actual_concepts: list) -> dict:
    """Evaluate how well the predicted trace matches actual internal concepts."""
    
    # Extract tokens from predicted trace
    predicted_tokens = set(t.strip().lower() for t in predicted.replace("→", " ").split())
    
    # Get actual concept tokens
    actual_tokens = set(c[0].strip().lower() for c in actual_concepts)
    
    # Calculate overlap
    overlap = predicted_tokens & actual_tokens
    
    precision = len(overlap) / len(predicted_tokens) if predicted_tokens else 0
    recall = len(overlap) / len(actual_tokens) if actual_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "predicted_tokens": list(predicted_tokens),
        "actual_tokens": list(actual_tokens),
        "overlap": list(overlap),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    print("="*60)
    print("THE MONEY SHOT: Cross-Model Transfer Test")
    print("="*60)
    print("RTP trained on: Llama 3.1 8B traces")
    print("Testing on: Mistral 7B internals")
    print("="*60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Test prompts (completion style)
    test_prompts = [
        "The capital of Texas is",
        "Dallas is a city in the state of",
        "The largest city in California is",
        "Apple was founded by Steve",
        "The company that makes Windows is",
        "The capital of France is",
        "The author of Harry Potter is",
    ]
    
    # Load RTP
    print("\nLoading RTP (Llama-trained)...")
    rtp_model, tokenizer, _ = load_rtp()
    print("✓ RTP loaded")
    
    # Load Mistral
    print("\nLoading Mistral 7B...")
    mistral = HookedTransformer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        device=device,
        dtype=torch.float16,
    )
    print("✓ Mistral loaded")
    
    # Run the experiment
    print(f"\n{'='*60}")
    print("Running transfer test...")
    print("="*60)
    
    results = []
    
    for prompt in test_prompts:
        print(f"\n--- {prompt!r} ---")
        
        # Get Mistral's actual internal trace
        actual = get_mistral_trace(mistral, prompt)
        print(f"Mistral final output: {actual['final_output']!r}")
        print(f"Mistral internal: {actual['internal_concepts'][:3]}")
        
        # Get RTP prediction
        predicted = predict_trace(
            rtp_model, tokenizer, 
            prompt, actual['final_output'], 
            device
        )
        print(f"RTP predicted: {predicted!r}")
        
        # Evaluate
        metrics = evaluate_overlap(predicted, actual['internal_concepts'])
        print(f"Overlap: {metrics['overlap']} (F1={metrics['f1']:.2f})")
        
        results.append({
            "prompt": prompt,
            "mistral_output": actual["final_output"],
            "mistral_concepts": actual["internal_concepts"],
            "rtp_prediction": predicted,
            "metrics": metrics
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    
    avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)
    avg_precision = sum(r["metrics"]["precision"] for r in results) / len(results)
    avg_recall = sum(r["metrics"]["recall"] for r in results) / len(results)
    
    print(f"\nAverage F1:        {avg_f1:.2%}")
    print(f"Average Precision: {avg_precision:.2%}")
    print(f"Average Recall:    {avg_recall:.2%}")
    
    # Interpretation
    print(f"\n{'='*60}")
    if avg_f1 > 0.5:
        print("🎉 STRONG TRANSFER! >50% overlap — this is publishable!")
    elif avg_f1 > 0.2:
        print("📝 PARTIAL TRANSFER. 20-50% overlap — workshop paper material")
    else:
        print("📊 LIMITED TRANSFER. <20% — analyze failure modes")
    print("="*60)
    
    # Save results
    output_path = Path("experiments/cross_model_transfer_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "avg_f1": avg_f1,
                "avg_precision": avg_precision, 
                "avg_recall": avg_recall
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
