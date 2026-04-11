#!/usr/bin/env python3
"""
Validation Experiment C: Representational Geometry Direct Test

Test whether semantic entity V vectors and numerical V vectors actually
occupy different regions of representation space.

Design:
- Extract 50 clean V vectors from semantic entity questions
- Extract 50 clean V vectors from numerical questions
- Compute pairwise cosine similarities:
  - Within semantic-entity group
  - Within numerical group
  - Between groups
- If within-group > between-group, direct evidence for manifold separation
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import sys


def make_prompt(question: str) -> str:
    return f"Question: {question}\n\nAnswer:"


class VectorExtractor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def get_v_vector(self, prompt: str, layer: int = 13):
        """Extract V vector at specified layer."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True, return_dict=True)
        kv = outputs.past_key_values
        
        if hasattr(kv, 'value_cache'):
            v = kv.value_cache[layer]
        elif hasattr(kv, '__getitem__'):
            try:
                v = kv[layer][1]
            except TypeError:
                # DynamicCache - access differently
                v = list(kv)[layer][1]
        else:
            # Fall back to iteration
            for i, layer_kv in enumerate(kv):
                if i == layer:
                    v = layer_kv[1]
                    break
        
        # Flatten to single vector (mean over positions and heads)
        v_flat = v.squeeze(0).mean(dim=(0, 1)).float().cpu().numpy()
        return v_flat


# Semantic entity questions (answers are names/entities)
ENTITY_QUESTIONS = [
    "Who was the first president of the United States?",
    "Who wrote Romeo and Juliet?",
    "Who painted the Mona Lisa?",
    "Who discovered penicillin?",
    "Who was the first person to walk on the moon?",
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of Germany?",
    "What is the largest ocean?",
    "What is the longest river in the world?",
    "Who wrote Pride and Prejudice?",
    "Who composed the Four Seasons?",
    "What is the tallest mountain in the world?",
    "What is the largest country by area?",
    "Who invented the telephone?",
    "What is the chemical symbol for gold?",
    "What planet is known as the Red Planet?",
    "Who painted Starry Night?",
    "What is the capital of Australia?",
    "Who wrote 1984?",
    "What is the largest mammal?",
    "Who discovered America?",
    "What is the capital of Italy?",
    "Who wrote The Great Gatsby?",
    "What is the smallest country in the world?",
    "Who invented the light bulb?",
    "What is the capital of Spain?",
    "Who wrote Hamlet?",
    "What is the largest desert?",
    "Who discovered gravity?",
    "What is the capital of Brazil?",
    "Who painted the Sistine Chapel ceiling?",
    "What is the deepest ocean?",
    "Who wrote Don Quixote?",
    "What is the capital of Canada?",
    "Who invented the printing press?",
    "What is the longest wall in the world?",
    "Who wrote War and Peace?",
    "What is the capital of China?",
    "Who discovered electricity?",
    "What is the largest lake?",
    "Who wrote The Odyssey?",
    "What is the capital of India?",
    "Who invented the airplane?",
    "What is the tallest building in the world?",
    "Who wrote Crime and Punishment?",
    "What is the capital of Russia?",
    "Who discovered X-rays?",
    "What is the largest island?",
    "Who wrote Les Misérables?",
]

# Numerical questions (answers are numbers)
NUMERICAL_QUESTIONS = [
    "What is 7 times 8?",
    "What is 12 times 12?",
    "What is 9 times 9?",
    "What is 15 times 3?",
    "What is 25 times 4?",
    "What is the square root of 144?",
    "What is the square root of 100?",
    "What is the square root of 81?",
    "What is the square root of 64?",
    "What is the square root of 49?",
    "In what year did World War II end?",
    "In what year did World War I begin?",
    "In what year did the Berlin Wall fall?",
    "In what year did humans land on the moon?",
    "In what year did Columbus reach America?",
    "What is 23 plus 47?",
    "What is 56 minus 29?",
    "What is 8 times 7?",
    "What is 6 times 9?",
    "What is 11 times 11?",
    "How many continents are there?",
    "How many planets are in our solar system?",
    "How many days are in a year?",
    "How many hours are in a day?",
    "How many minutes are in an hour?",
    "What is 100 divided by 5?",
    "What is 144 divided by 12?",
    "What is 81 divided by 9?",
    "What is 64 divided by 8?",
    "What is 49 divided by 7?",
    "In what year was the Declaration of Independence signed?",
    "In what year did the French Revolution begin?",
    "In what year did the American Civil War end?",
    "In what year was the Constitution ratified?",
    "In what year did the Industrial Revolution begin?",
    "What is 13 times 7?",
    "What is 17 times 6?",
    "What is 19 times 5?",
    "What is 23 times 4?",
    "What is 29 times 3?",
    "What is the square root of 169?",
    "What is the square root of 196?",
    "What is the square root of 225?",
    "What is the square root of 256?",
    "What is the square root of 289?",
    "How many weeks are in a year?",
    "How many months are in a year?",
    "How many seconds are in a minute?",
    "How many ounces are in a pound?",
    "How many feet are in a mile?",
]


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def run_validation(model_name: str = "google/gemma-4-E2B", n_per_group: int = 50):
    print(f"=" * 70, flush=True)
    print("VALIDATION EXPERIMENT C: Representational Geometry", flush=True)
    print(f"n = {n_per_group} per group", flush=True)
    print(f"=" * 70, flush=True)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    print(f"\nLoading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    
    extractor = VectorExtractor(model, tokenizer)
    
    # Extract V vectors
    print(f"\n{'-' * 70}", flush=True)
    print("EXTRACTING V VECTORS", flush=True)
    print(f"{'-' * 70}", flush=True)
    
    entity_vectors = []
    for i, q in enumerate(ENTITY_QUESTIONS[:n_per_group]):
        prompt = make_prompt(q)
        v = extractor.get_v_vector(prompt)
        entity_vectors.append(v)
        if (i + 1) % 10 == 0:
            print(f"  Entity: {i+1}/{n_per_group}", flush=True)
    
    numerical_vectors = []
    for i, q in enumerate(NUMERICAL_QUESTIONS[:n_per_group]):
        prompt = make_prompt(q)
        v = extractor.get_v_vector(prompt)
        numerical_vectors.append(v)
        if (i + 1) % 10 == 0:
            print(f"  Numerical: {i+1}/{n_per_group}", flush=True)
    
    # Compute similarities
    print(f"\n{'-' * 70}", flush=True)
    print("COMPUTING SIMILARITIES", flush=True)
    print(f"{'-' * 70}", flush=True)
    
    # Within entity group
    entity_sims = []
    for i in range(len(entity_vectors)):
        for j in range(i + 1, len(entity_vectors)):
            sim = cosine_similarity(entity_vectors[i], entity_vectors[j])
            entity_sims.append(sim)
    
    # Within numerical group
    numerical_sims = []
    for i in range(len(numerical_vectors)):
        for j in range(i + 1, len(numerical_vectors)):
            sim = cosine_similarity(numerical_vectors[i], numerical_vectors[j])
            numerical_sims.append(sim)
    
    # Between groups
    between_sims = []
    for ev in entity_vectors:
        for nv in numerical_vectors:
            sim = cosine_similarity(ev, nv)
            between_sims.append(sim)
    
    # Statistics
    entity_mean = np.mean(entity_sims)
    entity_std = np.std(entity_sims)
    numerical_mean = np.mean(numerical_sims)
    numerical_std = np.std(numerical_sims)
    between_mean = np.mean(between_sims)
    between_std = np.std(between_sims)
    
    # Results
    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    
    print(f"\nWithin-Entity Similarity:", flush=True)
    print(f"  Mean: {entity_mean:.4f} ± {entity_std:.4f}", flush=True)
    print(f"  n pairs: {len(entity_sims)}", flush=True)
    
    print(f"\nWithin-Numerical Similarity:", flush=True)
    print(f"  Mean: {numerical_mean:.4f} ± {numerical_std:.4f}", flush=True)
    print(f"  n pairs: {len(numerical_sims)}", flush=True)
    
    print(f"\nBetween-Group Similarity:", flush=True)
    print(f"  Mean: {between_mean:.4f} ± {between_std:.4f}", flush=True)
    print(f"  n pairs: {len(between_sims)}", flush=True)
    
    # Interpretation
    print(f"\n{'-' * 70}", flush=True)
    print("INTERPRETATION", flush=True)
    print(f"{'-' * 70}", flush=True)
    
    within_avg = (entity_mean + numerical_mean) / 2
    separation = within_avg - between_mean
    
    print(f"\nWithin-group average: {within_avg:.4f}", flush=True)
    print(f"Between-group average: {between_mean:.4f}", flush=True)
    print(f"Separation: {separation:.4f}", flush=True)
    
    if separation > 0.05:
        print(f"\n✅ Clear geometric separation — within-group > between-group", flush=True)
        interpretation = "separated"
    elif separation > 0:
        print(f"\n📊 Weak separation — within-group slightly > between-group", flush=True)
        interpretation = "weak_separation"
    else:
        print(f"\n❌ No separation — groups not geometrically distinct", flush=True)
        interpretation = "no_separation"
    
    # Save results
    output = {
        'experiment': 'C_geometry',
        'n_per_group': n_per_group,
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'results': {
            'entity': {
                'mean': float(entity_mean),
                'std': float(entity_std),
                'n_pairs': len(entity_sims),
            },
            'numerical': {
                'mean': float(numerical_mean),
                'std': float(numerical_std),
                'n_pairs': len(numerical_sims),
            },
            'between': {
                'mean': float(between_mean),
                'std': float(between_std),
                'n_pairs': len(between_sims),
            },
        },
        'within_avg': float(within_avg),
        'between_avg': float(between_mean),
        'separation': float(separation),
        'interpretation': interpretation,
    }
    
    results_dir = Path(__file__).parent.parent / "data" / "results"
    output_path = results_dir / "26_validation_geometry.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\nResults saved to: {output_path}", flush=True)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    return output


if __name__ == "__main__":
    run_validation()
