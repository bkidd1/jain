#!/usr/bin/env python3
"""
Same-Topic Contamination - n=100

Scale up to get confidence intervals on the contamination effect.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
import argparse
import random

# Expanded question set - all tricky capitals
QUESTIONS = [
    {"q": "What is the capital of Australia?", "correct": "Canberra", "wrong": "Sydney"},
    {"q": "What is the capital of Brazil?", "correct": "Brasilia", "wrong": "Rio de Janeiro"},
    {"q": "What is the capital of Turkey?", "correct": "Ankara", "wrong": "Istanbul"},
    {"q": "What is the capital of Switzerland?", "correct": "Bern", "wrong": "Zurich"},
    {"q": "What is the capital of Myanmar?", "correct": "Naypyidaw", "wrong": "Yangon"},
    {"q": "What is the capital of Nigeria?", "correct": "Abuja", "wrong": "Lagos"},
    {"q": "What is the capital of Vietnam?", "correct": "Hanoi", "wrong": "Ho Chi Minh City"},
    {"q": "What is the capital of New Zealand?", "correct": "Wellington", "wrong": "Auckland"},
    {"q": "What is the capital of Kazakhstan?", "correct": "Astana", "wrong": "Almaty"},
    {"q": "What is the capital of Morocco?", "correct": "Rabat", "wrong": "Casablanca"},
    {"q": "What is the capital of South Africa?", "correct": "Pretoria", "wrong": "Johannesburg"},
    {"q": "What is the capital of Canada?", "correct": "Ottawa", "wrong": "Toronto"},
    {"q": "What is the capital of India?", "correct": "New Delhi", "wrong": "Mumbai"},
    {"q": "What is the capital of China?", "correct": "Beijing", "wrong": "Shanghai"},
    {"q": "What is the capital of Pakistan?", "correct": "Islamabad", "wrong": "Karachi"},
    {"q": "What is the capital of Philippines?", "correct": "Manila", "wrong": "Cebu"},
    {"q": "What is the capital of Sri Lanka?", "correct": "Colombo", "wrong": "Kandy"},
    {"q": "What is the capital of Tanzania?", "correct": "Dodoma", "wrong": "Dar es Salaam"},
    {"q": "What is the capital of Ivory Coast?", "correct": "Yamoussoukro", "wrong": "Abidjan"},
    {"q": "What is the capital of Israel?", "correct": "Jerusalem", "wrong": "Tel Aviv"},
    {"q": "What is the capital of Bolivia?", "correct": "Sucre", "wrong": "La Paz"},
    {"q": "What is the capital of Benin?", "correct": "Porto-Novo", "wrong": "Cotonou"},
    {"q": "What is the capital of Montenegro?", "correct": "Podgorica", "wrong": "Cetinje"},
    {"q": "What is the capital of Palau?", "correct": "Ngerulmud", "wrong": "Koror"},
    {"q": "What is the capital of Malaysia?", "correct": "Kuala Lumpur", "wrong": "Putrajaya"},
    {"q": "What is the capital of Ecuador?", "correct": "Quito", "wrong": "Guayaquil"},
    {"q": "What is the capital of Cameroon?", "correct": "Yaounde", "wrong": "Douala"},
    {"q": "What is the capital of Belize?", "correct": "Belmopan", "wrong": "Belize City"},
    {"q": "What is the capital of Madagascar?", "correct": "Antananarivo", "wrong": "Toamasina"},
    {"q": "What is the capital of Mozambique?", "correct": "Maputo", "wrong": "Beira"},
    {"q": "What is the capital of Senegal?", "correct": "Dakar", "wrong": "Thies"},
    {"q": "What is the capital of Ghana?", "correct": "Accra", "wrong": "Kumasi"},
    {"q": "What is the capital of Kenya?", "correct": "Nairobi", "wrong": "Mombasa"},
    {"q": "What is the capital of Uganda?", "correct": "Kampala", "wrong": "Entebbe"},
    {"q": "What is the capital of Ethiopia?", "correct": "Addis Ababa", "wrong": "Dire Dawa"},
    {"q": "What is the capital of Egypt?", "correct": "Cairo", "wrong": "Alexandria"},
    {"q": "What is the capital of Peru?", "correct": "Lima", "wrong": "Cusco"},
    {"q": "What is the capital of Chile?", "correct": "Santiago", "wrong": "Valparaiso"},
    {"q": "What is the capital of Colombia?", "correct": "Bogota", "wrong": "Medellin"},
    {"q": "What is the capital of Venezuela?", "correct": "Caracas", "wrong": "Maracaibo"},
    {"q": "What is the capital of Argentina?", "correct": "Buenos Aires", "wrong": "Cordoba"},
    {"q": "What is the capital of Thailand?", "correct": "Bangkok", "wrong": "Chiang Mai"},
    {"q": "What is the capital of Indonesia?", "correct": "Jakarta", "wrong": "Surabaya"},
    {"q": "What is the capital of South Korea?", "correct": "Seoul", "wrong": "Busan"},
    {"q": "What is the capital of Japan?", "correct": "Tokyo", "wrong": "Osaka"},
    {"q": "What is the capital of Taiwan?", "correct": "Taipei", "wrong": "Kaohsiung"},
    {"q": "What is the capital of Bangladesh?", "correct": "Dhaka", "wrong": "Chittagong"},
    {"q": "What is the capital of Nepal?", "correct": "Kathmandu", "wrong": "Pokhara"},
    {"q": "What is the capital of Cambodia?", "correct": "Phnom Penh", "wrong": "Siem Reap"},
    {"q": "What is the capital of Laos?", "correct": "Vientiane", "wrong": "Luang Prabang"},
]


def classify(response: str, correct: str, wrong: str) -> Literal["correct", "wrong", "unclear"]:
    response_lower = response.lower()
    has_correct = correct.lower() in response_lower
    has_wrong = wrong.lower() in response_lower
    
    if has_correct and not has_wrong:
        return "correct"
    elif has_wrong and not has_correct:
        return "wrong"
    elif has_correct and has_wrong:
        return "correct" if response_lower.find(correct.lower()) < response_lower.find(wrong.lower()) else "wrong"
    return "unclear"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 30) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval."""
    if n == 0:
        return (0, 0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * ((p*(1-p)/n + z**2/(4*n**2)) ** 0.5) / denom
    return (max(0, center - spread), min(1, center + spread))


def run_test(model, tokenizer, n: int = 100):
    # Sample n questions (with replacement if needed)
    if n <= len(QUESTIONS):
        questions = random.sample(QUESTIONS, n)
    else:
        questions = [random.choice(QUESTIONS) for _ in range(n)]
    
    results = {"baseline": [], "prior_wrong": [], "prior_correct": []}
    
    # Condition 1: Baseline
    print(f"\n{'='*60}")
    print(f"CONDITION 1: Baseline (n={n})")
    print(f"{'='*60}")
    
    for i, q in enumerate(questions):
        prompt = f"Question: {q['q']}\n\nAnswer:"
        response = generate(model, tokenizer, prompt)
        classification = classify(response, q['correct'], q['wrong'])
        results["baseline"].append(classification)
        
        if (i + 1) % 20 == 0:
            correct = sum(1 for c in results["baseline"] if c == "correct")
            print(f"  [{i+1}/{n}] {correct}/{i+1} correct")
    
    # Condition 2: Prior wrong context
    print(f"\n{'='*60}")
    print(f"CONDITION 2: Prior wrong context (n={n})")
    print(f"{'='*60}")
    
    for i, q in enumerate(questions):
        first_turn = f"Question: {q['q']}\n\nAnswer: {q['wrong']}."
        second_turn = f"\n\nQuestion: {q['q']}\n\nAnswer:"
        prompt = first_turn + second_turn
        
        response = generate(model, tokenizer, prompt)
        classification = classify(response, q['correct'], q['wrong'])
        results["prior_wrong"].append(classification)
        
        if (i + 1) % 20 == 0:
            wrong = sum(1 for c in results["prior_wrong"] if c == "wrong")
            print(f"  [{i+1}/{n}] {wrong}/{i+1} contaminated")
    
    # Condition 3: Prior correct context
    print(f"\n{'='*60}")
    print(f"CONDITION 3: Prior correct context (n={n})")
    print(f"{'='*60}")
    
    for i, q in enumerate(questions):
        first_turn = f"Question: {q['q']}\n\nAnswer: {q['correct']}."
        second_turn = f"\n\nQuestion: {q['q']}\n\nAnswer:"
        prompt = first_turn + second_turn
        
        response = generate(model, tokenizer, prompt)
        classification = classify(response, q['correct'], q['wrong'])
        results["prior_correct"].append(classification)
        
        if (i + 1) % 20 == 0:
            correct = sum(1 for c in results["prior_correct"] if c == "correct")
            print(f"  [{i+1}/{n}] {correct}/{i+1} correct")
    
    return results, questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    
    random.seed(42)
    
    print(f"Loading model: {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    results, questions = run_test(model, tokenizer, args.n)
    
    n = args.n
    
    baseline_correct = sum(1 for c in results["baseline"] if c == "correct")
    baseline_wrong = sum(1 for c in results["baseline"] if c == "wrong")
    
    prior_wrong_correct = sum(1 for c in results["prior_wrong"] if c == "correct")
    prior_wrong_wrong = sum(1 for c in results["prior_wrong"] if c == "wrong")
    
    prior_correct_correct = sum(1 for c in results["prior_correct"] if c == "correct")
    prior_correct_wrong = sum(1 for c in results["prior_correct"] if c == "wrong")
    
    # CIs for wrong answer rates
    baseline_wrong_ci = wilson_ci(baseline_wrong, n)
    prior_wrong_wrong_ci = wilson_ci(prior_wrong_wrong, n)
    prior_correct_wrong_ci = wilson_ci(prior_correct_wrong, n)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nBaseline:")
    print(f"  Correct: {baseline_correct}/{n} ({100*baseline_correct/n:.1f}%)")
    print(f"  Wrong:   {baseline_wrong}/{n} ({100*baseline_wrong/n:.1f}%) [95% CI: {100*baseline_wrong_ci[0]:.1f}-{100*baseline_wrong_ci[1]:.1f}%]")
    
    print(f"\nPrior wrong context:")
    print(f"  Correct: {prior_wrong_correct}/{n} ({100*prior_wrong_correct/n:.1f}%)")
    print(f"  Wrong:   {prior_wrong_wrong}/{n} ({100*prior_wrong_wrong/n:.1f}%) [95% CI: {100*prior_wrong_wrong_ci[0]:.1f}-{100*prior_wrong_wrong_ci[1]:.1f}%]")
    
    print(f"\nPrior correct context:")
    print(f"  Correct: {prior_correct_correct}/{n} ({100*prior_correct_correct/n:.1f}%)")
    print(f"  Wrong:   {prior_correct_wrong}/{n} ({100*prior_correct_wrong/n:.1f}%) [95% CI: {100*prior_correct_wrong_ci[0]:.1f}-{100*prior_correct_wrong_ci[1]:.1f}%]")
    
    contamination_effect = prior_wrong_wrong - baseline_wrong
    print(f"\n{'='*60}")
    print(f"CONTAMINATION EFFECT: +{contamination_effect} wrong answers (+{100*contamination_effect/n:.1f}pp)")
    print(f"{'='*60}")
    
    # Check if CIs overlap
    if prior_wrong_wrong_ci[0] > baseline_wrong_ci[1]:
        print("CIs DO NOT OVERLAP — statistically significant contamination effect")
    else:
        print("CIs OVERLAP — need more samples or effect is small")
    
    # Save
    output_dir = Path(__file__).parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "model": args.model,
        "n": n,
        "baseline": {
            "correct": baseline_correct,
            "wrong": baseline_wrong,
            "wrong_ci": baseline_wrong_ci,
        },
        "prior_wrong": {
            "correct": prior_wrong_correct,
            "wrong": prior_wrong_wrong,
            "wrong_ci": prior_wrong_wrong_ci,
        },
        "prior_correct": {
            "correct": prior_correct_correct,
            "wrong": prior_correct_wrong,
            "wrong_ci": prior_correct_wrong_ci,
        },
        "contamination_effect": contamination_effect,
        "results": results,
    }
    
    output_path = output_dir / "06_contamination_n100.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
