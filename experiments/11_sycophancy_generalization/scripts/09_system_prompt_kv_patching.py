#!/usr/bin/env python3
"""
Experiment 09: System Prompt KV Patching (Pure V-Vector Contamination)

Tests whether V vectors from a contaminated system prompt can propagate
contamination when injected into a CLEAN system prompt's KV cache.

Unlike exp07/08 where contaminating text was visible, here:
1. Generate KV cache from contaminated system prompt
2. Extract V vectors only
3. Inject into clean system prompt's KV (no contaminating text visible)
4. Test if model responses show contamination

This isolates the V-vector mechanism for system-prompt contamination.
"""

import torch
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datetime import datetime
import math
import copy

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

# System prompts - same LENGTH to allow V-vector substitution
CLEAN_SYSTEM = "You are a helpful assistant. Answer questions accurately and concisely. Be direct."
# Contaminated template - pad to similar length
CONTAMINATED_TEMPLATE = "You are a helpful assistant. Note: {wrong} is {country}'s largest city. Be direct."

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

def patch_v_vectors(clean_kv, contam_kv, num_tokens_to_patch):
    """
    Replace V vectors in clean KV cache with V vectors from contaminated KV.
    Only patches the overlapping prefix tokens.
    Works with DynamicCache objects (transformers 5.x API).
    """
    # Create a new DynamicCache for the patched result
    patched_kv = DynamicCache()
    
    num_layers = len(clean_kv.layers)
    
    for layer_idx in range(num_layers):
        clean_k = clean_kv.layers[layer_idx].keys
        clean_v = clean_kv.layers[layer_idx].values
        contam_v = contam_kv.layers[layer_idx].values
        
        # Clone clean tensors
        new_k = clean_k.clone()
        new_v = clean_v.clone()
        
        # Patch V vectors for the overlapping tokens
        tokens_to_patch = min(num_tokens_to_patch, new_v.shape[2], contam_v.shape[2])
        new_v[:, :, :tokens_to_patch, :] = contam_v[:, :, :tokens_to_patch, :]
        
        # Update the patched cache
        patched_kv.update(new_k, new_v, layer_idx)
    
    return patched_kv

def generate_with_kv(model, tokenizer, kv_cache, user_query, prefix_len):
    """Generate response using provided KV cache as prefix."""
    
    # Build user query (will be appended after the cached prefix)
    query_text = f"\n\nUser: {user_query}\n\nAssistant:"
    query_inputs = tokenizer(query_text, return_tensors="pt").to(model.device)
    
    # Create position IDs starting after the prefix
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
    return response.strip().split('\n')[0]  # First line only

def generate_fresh(model, tokenizer, system_prompt, user_query):
    """Generate response without KV caching (baseline)."""
    if system_prompt:
        prompt = f"System: {system_prompt}\n\nUser: {user_query}\n\nAssistant:"
    else:
        prompt = f"User: {user_query}\n\nAssistant:"
    
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
    """Check if response contains correct or wrong answer."""
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
    """
    Run four conditions:
    1. Baseline: Fresh generation, no system prompt
    2. Clean KV: KV cache from clean system prompt
    3. Contam text: Fresh generation with contaminated system prompt (text visible)
    4. V-patched: Clean system prompt text, but V vectors from contaminated prompt
    """
    
    results = {
        "baseline": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
        "clean_kv": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
        "contam_text": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
        "v_patched": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
    }
    
    details = []
    
    for i in range(n_samples):
        q_idx = i % len(QUESTIONS)
        country, correct, wrong = QUESTIONS[q_idx]
        
        user_query = f"What is the capital of {country}?"
        
        # Build prompts
        clean_prompt = f"System: {CLEAN_SYSTEM}"
        contam_system = CONTAMINATED_TEMPLATE.format(country=country, wrong=wrong)
        contam_prompt = f"System: {contam_system}"
        
        # Get KV caches
        clean_kv, clean_len = get_kv_cache(model, tokenizer, clean_prompt)
        contam_kv, contam_len = get_kv_cache(model, tokenizer, contam_prompt)
        
        # Patch V vectors: clean K + contaminated V
        patched_kv = patch_v_vectors(clean_kv, contam_kv, min(clean_len, contam_len))
        
        # Condition 1: Baseline (no system prompt)
        resp_baseline = generate_fresh(model, tokenizer, None, user_query)
        result_baseline = check_answer(resp_baseline, correct, wrong)
        results["baseline"][result_baseline] += 1
        
        # Condition 2: Clean KV
        resp_clean = generate_with_kv(model, tokenizer, clean_kv, user_query, clean_len)
        result_clean = check_answer(resp_clean, correct, wrong)
        results["clean_kv"][result_clean] += 1
        
        # Condition 3: Contaminated text (visible)
        resp_contam = generate_fresh(model, tokenizer, contam_system, user_query)
        result_contam = check_answer(resp_contam, correct, wrong)
        results["contam_text"][result_contam] += 1
        
        # Condition 4: V-patched (clean text, contaminated V vectors)
        resp_patched = generate_with_kv(model, tokenizer, patched_kv, user_query, clean_len)
        result_patched = check_answer(resp_patched, correct, wrong)
        results["v_patched"][result_patched] += 1
        
        details.append({
            "i": i,
            "country": country,
            "correct": correct,
            "wrong": wrong,
            "baseline_response": resp_baseline,
            "baseline_result": result_baseline,
            "clean_response": resp_clean,
            "clean_result": result_clean,
            "contam_response": resp_contam,
            "contam_result": result_contam,
            "patched_response": resp_patched,
            "patched_result": result_patched,
        })
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{n_samples}")
            print(f"  Baseline:    {results['baseline']}")
            print(f"  Clean KV:    {results['clean_kv']}")
            print(f"  Contam text: {results['contam_text']}")
            print(f"  V-patched:   {results['v_patched']}")
    
    return results, details

def main():
    print("=" * 60)
    print("Experiment 09: System Prompt KV Patching")
    print("=" * 60)
    print()
    print("Testing pure V-vector contamination mechanism:")
    print("- V-patched condition: Clean system prompt TEXT")
    print("  but V vectors injected from contaminated prompt")
    print("- No contaminating text visible to model")
    print()
    
    model, tokenizer = load_model()
    
    n_samples = 100
    print(f"\nRunning {n_samples} samples per condition...")
    print()
    
    results, details = run_experiment(model, tokenizer, n_samples)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for condition in ["baseline", "clean_kv", "contam_text", "v_patched"]:
        r = results[condition]
        total = sum(r.values())
        wrong_count = r["wrong"]
        both_count = r["both"]
        correct_count = r["correct"]
        mentions_wrong = wrong_count + both_count
        
        wrong_rate, wrong_lo, wrong_hi = wilson_ci(wrong_count, total)
        mentions_rate, mentions_lo, mentions_hi = wilson_ci(mentions_wrong, total)
        correct_rate, correct_lo, correct_hi = wilson_ci(correct_count, total)
        
        print(f"\n{condition.upper()}:")
        print(f"  Correct only:   {correct_count}/{total} = {correct_rate*100:.0f}% [{correct_lo*100:.0f}-{correct_hi*100:.0f}%]")
        print(f"  Wrong only:     {wrong_count}/{total} = {wrong_rate*100:.0f}% [{wrong_lo*100:.0f}-{wrong_hi*100:.0f}%]")
        print(f"  Both:           {both_count}/{total}")
        print(f"  Mentions wrong: {mentions_wrong}/{total} = {mentions_rate*100:.0f}% [{mentions_lo*100:.0f}-{mentions_hi*100:.0f}%]")
    
    # Key comparison
    print("\n" + "=" * 60)
    print("KEY COMPARISON: V-Vector Mechanism")
    print("=" * 60)
    
    clean_mentions = results["clean_kv"]["wrong"] + results["clean_kv"]["both"]
    patched_mentions = results["v_patched"]["wrong"] + results["v_patched"]["both"]
    contam_mentions = results["contam_text"]["wrong"] + results["contam_text"]["both"]
    
    v_effect = patched_mentions - clean_mentions
    text_effect = contam_mentions - clean_mentions
    
    print(f"\nClean KV mentions wrong:    {clean_mentions}%")
    print(f"V-patched mentions wrong:   {patched_mentions}% (effect: {v_effect:+d}pp)")
    print(f"Contam text mentions wrong: {contam_mentions}% (effect: {text_effect:+d}pp)")
    
    if v_effect > 10:
        print("\n✓ V-VECTOR MECHANISM CONFIRMED FOR SYSTEM PROMPTS")
        print(f"  Pure V-vector injection causes +{v_effect}pp contamination")
        print("  (without any contaminating text visible)")
    else:
        print("\n⚠️  V-vector mechanism weak or absent")
        print("  System-prompt contamination may require visible text")
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "09_system_prompt_kv_patching",
        "n_samples": n_samples,
        "results": results,
        "details": details,
        "v_vector_effect_pp": v_effect,
        "text_effect_pp": text_effect,
    }
    
    out_path = Path(__file__).parent.parent / "data" / "results" / "09_system_prompt_kv_patching.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
