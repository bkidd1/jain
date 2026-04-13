#!/usr/bin/env python3
"""
Experiment 10: Bidirectional Causality Test

Tests whether injecting CLEAN V vectors can REDUCE error rates
in an already-contaminated KV state.

Conditions:
1. Contaminated KV (full) - baseline contamination
2. V-sanitized (clean V, contaminated K) - tests if clean V cures
3. Clean KV (full) - best case control
4. Fresh baseline (no KV) - reference

If clean V injection reduces errors, it confirms V vectors are the causal locus.
"""

import torch
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datetime import datetime
import math

def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return p, max(0, center - spread), min(1, center + spread)

QUESTIONS = [
    ("Australia", "Canberra", "Sydney"),
    ("Brazil", "Brasília", "São Paulo"),
    ("Canada", "Ottawa", "Toronto"),
    ("China", "Beijing", "Shanghai"),
    ("India", "New Delhi", "Mumbai"),
    ("Myanmar", "Naypyidaw", "Yangon"),
    ("Nigeria", "Abuja", "Lagos"),
    ("Pakistan", "Islamabad", "Karachi"),
    ("South Africa", "Pretoria", "Johannesburg"),
    ("Switzerland", "Bern", "Zurich"),
    ("Tanzania", "Dodoma", "Dar es Salaam"),
    ("Turkey", "Ankara", "Istanbul"),
    ("United States", "Washington, D.C.", "New York"),
    ("Vietnam", "Hanoi", "Ho Chi Minh City"),
    ("Morocco", "Rabat", "Casablanca"),
    ("Ivory Coast", "Yamoussoukro", "Abidjan"),
    ("Kazakhstan", "Astana", "Almaty"),
    ("Sri Lanka", "Sri Jayawardenepura Kotte", "Colombo"),
    ("Belize", "Belmopan", "Belize City"),
    ("Bolivia", "Sucre", "La Paz"),
    ("Ecuador", "Quito", "Guayaquil"),
    ("Philippines", "Manila", "Quezon City"),
    ("New Zealand", "Wellington", "Auckland"),
    ("Israel", "Jerusalem", "Tel Aviv"),
    ("Malawi", "Lilongwe", "Blantyre"),
]

def load_model():
    print("Loading Qwen2.5-3B-Instruct...")
    start = time.time()
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loaded {model_id} in {time.time()-start:.1f}s")
    return model, tokenizer

def get_kv_cache(model, tokenizer, text):
    """Get KV cache for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )
    
    return outputs.past_key_values, inputs.input_ids.shape[1]

def create_patched_kv(source_k_kv, source_v_kv, num_layers):
    """Create KV cache with K from one source and V from another."""
    patched_kv = DynamicCache()
    
    for layer_idx in range(num_layers):
        k = source_k_kv.layers[layer_idx].keys.clone()
        v = source_v_kv.layers[layer_idx].values.clone()
        patched_kv.update(k, v, layer_idx)
    
    return patched_kv

def generate_with_kv(model, tokenizer, kv_cache, user_query, prefix_len):
    """Generate response using provided KV cache as prefix."""
    
    query_text = f"\n\nUser: {user_query}\n\nAssistant:"
    query_inputs = tokenizer(query_text, return_tensors="pt").to(model.device)
    
    position_ids = torch.arange(
        prefix_len, 
        prefix_len + query_inputs.input_ids.shape[1],
        device=model.device
    ).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=query_inputs.input_ids,
            attention_mask=torch.ones_like(query_inputs.input_ids),
            position_ids=position_ids,
            past_key_values=kv_cache,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][query_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip().split('\n')[0]

def generate_fresh(model, tokenizer, prompt):
    """Generate response without KV caching."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip().split('\n')[0]

def check_answer(response, correct, wrong):
    response_lower = response.lower()
    correct_lower = correct.lower()
    wrong_lower = wrong.lower()
    
    has_correct = correct_lower in response_lower
    has_wrong = wrong_lower in response_lower
    
    if has_correct and not has_wrong:
        return "correct"
    elif has_wrong and not has_correct:
        return "wrong"
    elif has_correct and has_wrong:
        return "both"
    else:
        return "neither"

def run_experiment(model, tokenizer, n_samples=100):
    results = {
        "baseline": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
        "contaminated_full": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
        "v_sanitized": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
        "clean_full": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
    }
    
    details = []
    num_layers = None
    
    for i in range(n_samples):
        q_idx = i % len(QUESTIONS)
        country, correct, wrong = QUESTIONS[q_idx]
        
        user_query = f"What is the capital of {country}?"
        
        # Build matched-length prompts for RoPE alignment
        # Contaminated: user says wrong answer
        contam_prompt = f"User: I'm pretty sure {wrong} is the capital of {country}, right?\n\nAssistant: Yes, that's correct! {wrong} is indeed the capital of {country}."
        # Clean: user says correct answer
        clean_prompt = f"User: I'm pretty sure {correct} is the capital of {country}, right?\n\nAssistant: Yes, that's correct! {correct} is indeed the capital of {country}."
        
        # Pad to same length
        contam_tokens = tokenizer(contam_prompt, return_tensors="pt")
        clean_tokens = tokenizer(clean_prompt, return_tensors="pt")
        max_len = max(contam_tokens.input_ids.shape[1], clean_tokens.input_ids.shape[1])
        
        # Pad shorter one
        if contam_tokens.input_ids.shape[1] < max_len:
            pad_len = max_len - contam_tokens.input_ids.shape[1]
            contam_prompt = " " * pad_len + contam_prompt
        elif clean_tokens.input_ids.shape[1] < max_len:
            pad_len = max_len - clean_tokens.input_ids.shape[1]
            clean_prompt = " " * pad_len + clean_prompt
        
        # Get KV caches
        contam_kv, contam_len = get_kv_cache(model, tokenizer, contam_prompt)
        clean_kv, clean_len = get_kv_cache(model, tokenizer, clean_prompt)
        
        if num_layers is None:
            num_layers = len(contam_kv.layers)
        
        # Create V-sanitized: contaminated K, clean V
        v_sanitized_kv = create_patched_kv(contam_kv, clean_kv, num_layers)
        
        # Condition 1: Fresh baseline
        baseline_prompt = f"User: {user_query}\n\nAssistant:"
        resp_baseline = generate_fresh(model, tokenizer, baseline_prompt)
        result_baseline = check_answer(resp_baseline, correct, wrong)
        results["baseline"][result_baseline] += 1
        
        # Condition 2: Contaminated full KV
        resp_contam = generate_with_kv(model, tokenizer, contam_kv, user_query, contam_len)
        result_contam = check_answer(resp_contam, correct, wrong)
        results["contaminated_full"][result_contam] += 1
        
        # Condition 3: V-sanitized (clean V, contam K)
        resp_sanitized = generate_with_kv(model, tokenizer, v_sanitized_kv, user_query, contam_len)
        result_sanitized = check_answer(resp_sanitized, correct, wrong)
        results["v_sanitized"][result_sanitized] += 1
        
        # Condition 4: Clean full KV
        resp_clean = generate_with_kv(model, tokenizer, clean_kv, user_query, clean_len)
        result_clean = check_answer(resp_clean, correct, wrong)
        results["clean_full"][result_clean] += 1
        
        details.append({
            "i": i,
            "country": country,
            "correct": correct,
            "wrong": wrong,
            "baseline_response": resp_baseline,
            "baseline_result": result_baseline,
            "contam_response": resp_contam,
            "contam_result": result_contam,
            "sanitized_response": resp_sanitized,
            "sanitized_result": result_sanitized,
            "clean_response": resp_clean,
            "clean_result": result_clean,
        })
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{n_samples}")
            print(f"  Baseline:     {results['baseline']}")
            print(f"  Contam full:  {results['contaminated_full']}")
            print(f"  V-sanitized:  {results['v_sanitized']}")
            print(f"  Clean full:   {results['clean_full']}")
    
    return results, details

def main():
    print("=" * 60)
    print("Experiment 10: Bidirectional Causality Test")
    print("=" * 60)
    print()
    print("Testing whether clean V vectors can CURE contamination")
    print()
    
    model, tokenizer = load_model()
    
    n_samples = 100
    print(f"\nRunning {n_samples} samples per condition...")
    print()
    
    results, details = run_experiment(model, tokenizer, n_samples)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for condition in ["baseline", "contaminated_full", "v_sanitized", "clean_full"]:
        r = results[condition]
        total = sum(r.values())
        correct = r["correct"]
        wrong = r["wrong"]
        
        correct_rate, correct_lo, correct_hi = wilson_ci(correct, total)
        wrong_rate, wrong_lo, wrong_hi = wilson_ci(wrong, total)
        
        print(f"\n{condition.upper()}:")
        print(f"  Correct: {correct}/{total} = {correct_rate*100:.0f}% [{correct_lo*100:.0f}-{correct_hi*100:.0f}%]")
        print(f"  Wrong:   {wrong}/{total} = {wrong_rate*100:.0f}% [{wrong_lo*100:.0f}-{wrong_hi*100:.0f}%]")
    
    # Key comparison
    print("\n" + "=" * 60)
    print("BIDIRECTIONAL CAUSALITY TEST")
    print("=" * 60)
    
    contam_wrong = results["contaminated_full"]["wrong"]
    sanitized_wrong = results["v_sanitized"]["wrong"]
    clean_wrong = results["clean_full"]["wrong"]
    baseline_wrong = results["baseline"]["wrong"]
    
    recovery_effect = contam_wrong - sanitized_wrong
    
    print(f"\nBaseline wrong:           {baseline_wrong}%")
    print(f"Contaminated full wrong:  {contam_wrong}%")
    print(f"V-sanitized wrong:        {sanitized_wrong}% (recovery: {recovery_effect:+d}pp)")
    print(f"Clean full wrong:         {clean_wrong}%")
    
    if recovery_effect > 20:
        print("\n✓ BIDIRECTIONAL CAUSALITY CONFIRMED")
        print(f"  Clean V vectors reduce errors by {recovery_effect}pp")
        print("  V vectors are causally responsible for contamination")
    else:
        print(f"\n⚠️  Recovery effect weak: {recovery_effect}pp")
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "10_bidirectional_causality",
        "n_samples": n_samples,
        "results": results,
        "details": details,
        "recovery_effect_pp": recovery_effect,
    }
    
    out_path = Path(__file__).parent.parent / "data" / "results" / "10_bidirectional_causality.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
