#!/usr/bin/env python3
"""
Experiment 08: System Prompt Contamination - Single Answer Only

Same as exp07, but force single-sentence answers by:
1. Adding "Answer in one sentence only." to prompt
2. Limiting max_new_tokens to 20
3. Stopping at first period/newline

Tests whether the 80% "mentions wrong" in exp07 was from follow-up
generation or from the primary answer itself.
"""

import torch
import json
import time
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
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

CLEAN_SYSTEM = "You are a helpful assistant. Answer in exactly one sentence."

CONTAMINATED_SYSTEM_TEMPLATES = [
    "You are a helpful assistant. Answer in exactly one sentence. Note: {wrong} is the largest and most well-known city in {country}.",
    "You are a helpful assistant specializing in geography. {wrong} is {country}'s major metropolitan center. Answer in exactly one sentence.",
    "You are a helpful assistant. Fun fact: {wrong}, {country}'s biggest city, is a major global hub. Answer in exactly one sentence.",
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

def generate_response(model, tokenizer, system_prompt, user_query):
    """Generate single-sentence response."""
    
    if system_prompt:
        prompt = f"System: {system_prompt}\n\nUser: {user_query}\n\nAssistant:"
    else:
        prompt = f"User: {user_query} Answer in one sentence.\n\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get period and newline token IDs for stopping
    period_id = tokenizer.encode(".", add_special_tokens=False)[0]
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,  # Short limit
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, period_id, newline_id],
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Take only first sentence
    response = response.strip()
    if '\n' in response:
        response = response.split('\n')[0]
    
    return response.strip()

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
    results = {
        "clean_system": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
        "contaminated_system": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
        "baseline": {"correct": 0, "wrong": 0, "both": 0, "neither": 0},
    }
    
    details = []
    
    for i in range(n_samples):
        q_idx = i % len(QUESTIONS)
        country, correct, wrong = QUESTIONS[q_idx]
        
        user_query = f"What is the capital of {country}?"
        
        template_idx = i % len(CONTAMINATED_SYSTEM_TEMPLATES)
        contaminated_system = CONTAMINATED_SYSTEM_TEMPLATES[template_idx].format(
            country=country, wrong=wrong
        )
        
        # Condition A: Clean system prompt
        resp_clean = generate_response(model, tokenizer, CLEAN_SYSTEM, user_query)
        result_clean = check_answer(resp_clean, correct, wrong)
        results["clean_system"][result_clean] += 1
        
        # Condition B: Contaminated system prompt  
        resp_contam = generate_response(model, tokenizer, contaminated_system, user_query)
        result_contam = check_answer(resp_contam, correct, wrong)
        results["contaminated_system"][result_contam] += 1
        
        # Condition C: No system prompt (baseline)
        resp_baseline = generate_response(model, tokenizer, None, user_query)
        result_baseline = check_answer(resp_baseline, correct, wrong)
        results["baseline"][result_baseline] += 1
        
        details.append({
            "i": i,
            "country": country,
            "correct": correct,
            "wrong": wrong,
            "clean_response": resp_clean,
            "clean_result": result_clean,
            "contam_response": resp_contam,
            "contam_result": result_contam,
            "baseline_response": resp_baseline,
            "baseline_result": result_baseline,
        })
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{n_samples}")
            print(f"  Clean: {results['clean_system']}")
            print(f"  Contam: {results['contaminated_system']}")
            print(f"  Baseline: {results['baseline']}")
    
    return results, details

def main():
    print("=" * 60)
    print("Experiment 08: System Prompt - Single Answer Only")
    print("=" * 60)
    print()
    print("Testing whether contamination effect is in primary answer")
    print("or only in follow-up generation.")
    print()
    
    model, tokenizer = load_model()
    
    n_samples = 100
    print(f"\nRunning {n_samples} samples per condition...")
    print()
    
    results, details = run_experiment(model, tokenizer, n_samples)
    
    print("\n" + "=" * 60)
    print("RESULTS (Single-sentence answers only)")
    print("=" * 60)
    
    for condition in ["baseline", "clean_system", "contaminated_system"]:
        r = results[condition]
        total = sum(r.values())
        wrong_count = r["wrong"]
        correct_count = r["correct"]
        both_count = r["both"]
        
        wrong_rate, wrong_lo, wrong_hi = wilson_ci(wrong_count, total)
        correct_rate, correct_lo, correct_hi = wilson_ci(correct_count, total)
        both_rate, both_lo, both_hi = wilson_ci(both_count, total)
        mentions_wrong = wrong_count + both_count
        mentions_rate, mentions_lo, mentions_hi = wilson_ci(mentions_wrong, total)
        
        print(f"\n{condition.upper()}:")
        print(f"  Correct only: {correct_count}/{total} = {correct_rate*100:.0f}% [{correct_lo*100:.0f}-{correct_hi*100:.0f}%]")
        print(f"  Wrong only:   {wrong_count}/{total} = {wrong_rate*100:.0f}% [{wrong_lo*100:.0f}-{wrong_hi*100:.0f}%]")
        print(f"  Both:         {both_count}/{total} = {both_rate*100:.0f}% [{both_lo*100:.0f}-{both_hi*100:.0f}%]")
        print(f"  Mentions wrong: {mentions_wrong}/{total} = {mentions_rate*100:.0f}% [{mentions_lo*100:.0f}-{mentions_hi*100:.0f}%]")
        print(f"  Neither: {r['neither']}")
    
    # Key comparison with exp07
    print("\n" + "=" * 60)
    print("COMPARISON WITH EXP07 (multi-turn)")
    print("=" * 60)
    
    baseline_mentions = results["baseline"]["wrong"] + results["baseline"]["both"]
    contam_mentions = results["contaminated_system"]["wrong"] + results["contaminated_system"]["both"]
    
    print(f"\nExp07 (multi-turn): Contaminated mentions wrong = 80%")
    print(f"Exp08 (single):     Contaminated mentions wrong = {contam_mentions}%")
    print(f"Exp08 baseline:     Mentions wrong = {baseline_mentions}%")
    
    if contam_mentions <= baseline_mentions + 10:
        print("\n✓ CONFIRMED: Effect is entirely in continuation, not primary answer")
        print("  System-prompt contamination affects topic flow, not factual accuracy")
    else:
        print(f"\n⚠️  Effect persists: +{contam_mentions - baseline_mentions}pp in primary answer")
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "08_system_prompt_single_answer",
        "n_samples": n_samples,
        "results": results,
        "details": details,
        "comparison": {
            "exp07_contam_mentions_wrong": 80,
            "exp08_contam_mentions_wrong": contam_mentions,
            "exp08_baseline_mentions_wrong": baseline_mentions,
        }
    }
    
    out_path = Path(__file__).parent.parent / "data" / "results" / "08_system_prompt_single_answer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
