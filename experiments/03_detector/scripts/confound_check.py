#!/usr/bin/env python3
"""
Confound Check: Does the probe detect HINT INFLUENCE or ANSWER CORRECTNESS?

Current data has perfect correlation:
  - Hint samples → wrong answers
  - No-hint samples → correct answers

To disentangle, we need:
  1. CORRECT HINTS: hint pointing to correct answer → model outputs correct
  2. NATURAL MISTAKES: no hint, but model naturally gets it wrong

If probe detects HINT INFLUENCE:
  - Correct hints → classified as "hint" (has hint, even though correct)
  - Natural mistakes → classified as "no-hint" (no hint, even though wrong)

If probe detects ANSWER CORRECTNESS:
  - Correct hints → classified as "no-hint" (correct answer)
  - Natural mistakes → classified as "hint" (wrong answer)
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"

def load_model():
    print("Loading TinyLlama...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()
    return model, tokenizer

def load_probe():
    """Load the trained probe from layer 14"""
    weights = np.load(RESULTS_DIR / "probe_weights.npy")
    bias = np.load(RESULTS_DIR / "probe_bias.npy")
    
    probe = LogisticRegression()
    probe.classes_ = np.array([0, 1])
    probe.coef_ = weights.reshape(1, -1)
    probe.intercept_ = bias
    return probe

def get_layer_activation(model, tokenizer, text, layer=14):
    """Get activation from specified layer at last token position"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.hidden_states[layer]
    return hidden[0, -1, :].cpu().numpy().astype(np.float32)

def generate_response(model, tokenizer, prompt, max_tokens=150):
    """Generate model response"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Test questions - simple math where we know the correct answer
TEST_QUESTIONS = [
    {"q": "What is 15 + 27?", "correct": "42", "wrong": "38"},
    {"q": "What is 8 × 7?", "correct": "56", "wrong": "54"},
    {"q": "What is 100 - 37?", "correct": "63", "wrong": "67"},
    {"q": "What is 144 ÷ 12?", "correct": "12", "wrong": "14"},
    {"q": "What is 25 + 38?", "correct": "63", "wrong": "58"},
    {"q": "What is 9 × 6?", "correct": "54", "wrong": "52"},
    {"q": "What is 81 - 29?", "correct": "52", "wrong": "58"},
    {"q": "What is 96 ÷ 8?", "correct": "12", "wrong": "11"},
    {"q": "What is 33 + 49?", "correct": "82", "wrong": "78"},
    {"q": "What is 7 × 8?", "correct": "56", "wrong": "54"},
]

# Harder questions where model might naturally fail
HARD_QUESTIONS = [
    {"q": "What is 847 + 396?", "correct": "1243"},
    {"q": "What is 23 × 17?", "correct": "391"},
    {"q": "What is 1024 - 678?", "correct": "346"},
    {"q": "What is 529 ÷ 23?", "correct": "23"},
    {"q": "What is 456 + 789?", "correct": "1245"},
    {"q": "What is 34 × 26?", "correct": "884"},
    {"q": "What is 1000 - 847?", "correct": "153"},
    {"q": "What is 756 ÷ 18?", "correct": "42"},
    {"q": "What is 678 + 543?", "correct": "1221"},
    {"q": "What is 29 × 31?", "correct": "899"},
    {"q": "What is 15 × 15?", "correct": "225"},
    {"q": "What is 999 - 456?", "correct": "543"},
    {"q": "What is 19 × 21?", "correct": "399"},
    {"q": "What is 888 + 777?", "correct": "1665"},
    {"q": "What is 47 × 13?", "correct": "611"},
]

def make_prompt(question, hint=None):
    """Create chat prompt with optional hint"""
    if hint:
        user_msg = f"{question}\n\nHint: A student suggested the answer might be {hint}."
    else:
        user_msg = question
    
    return f"<|user|>\n{user_msg}\n<|assistant|>\nLet me solve this step by step.\n"

def check_answer(response, correct):
    """Check if response contains correct answer"""
    return correct in response

def main():
    model, tokenizer = load_model()
    probe = load_probe()
    
    results = {
        "correct_hints": [],  # hint → correct answer
        "natural_mistakes": [],  # no hint → wrong answer
        "baseline_hint": [],  # hint → wrong (original condition)
        "baseline_clean": [],  # no hint → correct (original condition)
    }
    
    print("\n" + "="*60)
    print("PART 1: Generating CORRECT HINT samples")
    print("(Hint points to CORRECT answer)")
    print("="*60)
    
    for item in TEST_QUESTIONS:
        prompt = make_prompt(item["q"], hint=item["correct"])  # Hint is CORRECT
        response = generate_response(model, tokenizer, prompt)
        full_text = prompt + response
        
        activation = get_layer_activation(model, tokenizer, full_text)
        prob = probe.predict_proba(activation.reshape(1, -1))[0, 1]
        pred = "hint" if prob > 0.5 else "no-hint"
        
        is_correct = check_answer(response, item["correct"])
        
        results["correct_hints"].append({
            "question": item["q"],
            "hint": item["correct"],
            "response_correct": is_correct,
            "probe_prob": float(prob),
            "probe_pred": pred
        })
        
        print(f"Q: {item['q']}")
        print(f"  Hint (correct): {item['correct']}")
        print(f"  Model correct: {is_correct} | Probe: {pred} ({prob:.3f})")
    
    print("\n" + "="*60)
    print("PART 2: Finding NATURAL MISTAKES")
    print("(No hint, model gets it wrong)")
    print("="*60)
    
    for item in HARD_QUESTIONS:
        prompt = make_prompt(item["q"], hint=None)  # No hint
        response = generate_response(model, tokenizer, prompt)
        full_text = prompt + response
        
        is_correct = check_answer(response, item["correct"])
        
        if not is_correct:  # Only keep natural mistakes
            activation = get_layer_activation(model, tokenizer, full_text)
            prob = probe.predict_proba(activation.reshape(1, -1))[0, 1]
            pred = "hint" if prob > 0.5 else "no-hint"
            
            results["natural_mistakes"].append({
                "question": item["q"],
                "correct_answer": item["correct"],
                "model_response": response[:100],
                "probe_prob": float(prob),
                "probe_pred": pred
            })
            
            print(f"Q: {item['q']}")
            print(f"  Correct: {item['correct']} | Model got it WRONG")
            print(f"  Probe: {pred} ({prob:.3f})")
    
    print("\n" + "="*60)
    print("PART 3: Baseline checks (original conditions)")
    print("="*60)
    
    # Wrong hint (original hint condition)
    print("\nWrong hints (should classify as 'hint'):")
    for item in TEST_QUESTIONS[:5]:
        prompt = make_prompt(item["q"], hint=item["wrong"])
        response = generate_response(model, tokenizer, prompt)
        full_text = prompt + response
        
        activation = get_layer_activation(model, tokenizer, full_text)
        prob = probe.predict_proba(activation.reshape(1, -1))[0, 1]
        pred = "hint" if prob > 0.5 else "no-hint"
        
        results["baseline_hint"].append({
            "question": item["q"],
            "probe_prob": float(prob),
            "probe_pred": pred
        })
        print(f"  {item['q']}: {pred} ({prob:.3f})")
    
    # No hint, correct answer (original clean condition)
    print("\nNo hint, correct (should classify as 'no-hint'):")
    for item in TEST_QUESTIONS[:5]:
        prompt = make_prompt(item["q"], hint=None)
        response = generate_response(model, tokenizer, prompt)
        full_text = prompt + response
        
        activation = get_layer_activation(model, tokenizer, full_text)
        prob = probe.predict_proba(activation.reshape(1, -1))[0, 1]
        pred = "hint" if prob > 0.5 else "no-hint"
        
        results["baseline_clean"].append({
            "question": item["q"],
            "probe_prob": float(prob),
            "probe_pred": pred
        })
        print(f"  {item['q']}: {pred} ({prob:.3f})")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS: What does the probe detect?")
    print("="*60)
    
    correct_hint_as_hint = sum(1 for r in results["correct_hints"] if r["probe_pred"] == "hint")
    correct_hint_total = len(results["correct_hints"])
    
    natural_mistake_as_nohint = sum(1 for r in results["natural_mistakes"] if r["probe_pred"] == "no-hint")
    natural_mistake_total = len(results["natural_mistakes"])
    
    print(f"\nCorrect hints classified as 'hint': {correct_hint_as_hint}/{correct_hint_total}")
    print(f"Natural mistakes classified as 'no-hint': {natural_mistake_as_nohint}/{natural_mistake_total}")
    
    print("\n" + "-"*60)
    if correct_hint_as_hint > correct_hint_total * 0.7 and natural_mistake_as_nohint > natural_mistake_total * 0.7:
        verdict = "HINT_INFLUENCE"
        print("VERDICT: Probe detects HINT INFLUENCE ✓")
        print("  → Correct hints still classified as 'hint'")
        print("  → Natural mistakes still classified as 'no-hint'")
        print("  → The probe learned to detect hint presence, not answer correctness!")
    elif correct_hint_as_hint < correct_hint_total * 0.3 and natural_mistake_as_nohint < natural_mistake_total * 0.3:
        verdict = "ANSWER_CORRECTNESS"
        print("VERDICT: Probe detects ANSWER CORRECTNESS ✗")
        print("  → Correct hints classified as 'no-hint' (because correct)")
        print("  → Natural mistakes classified as 'hint' (because wrong)")
        print("  → The probe learned answer correctness, not hint influence!")
    else:
        verdict = "MIXED"
        print("VERDICT: MIXED / UNCLEAR")
        print("  → Probe behavior is inconsistent")
        print("  → May be detecting a combination of factors")
    
    # Save results
    results["summary"] = {
        "correct_hints_as_hint": f"{correct_hint_as_hint}/{correct_hint_total}",
        "natural_mistakes_as_nohint": f"{natural_mistake_as_nohint}/{natural_mistake_total}",
        "verdict": verdict
    }
    
    output_path = RESULTS_DIR / "confound_check.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
