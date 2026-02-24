#!/usr/bin/env python3
"""
Evaluation v2: Proper metrics for reasoning trace prediction.

Fixes from critique:
- Token F1 → BERTScore (semantic similarity)
- Zero ablation → Mean ablation
- Add logit difference metric
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from bert_score import score as bert_score
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Evaluation results with multiple metrics."""
    bert_precision: float
    bert_recall: float
    bert_f1: float
    exact_match: float
    semantic_overlap: float  # Proportion of concepts semantically matched
    

def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli"
) -> Tuple[float, float, float]:
    """
    Compute BERTScore between predictions and references.
    
    Returns: (precision, recall, f1)
    """
    P, R, F1 = bert_score(
        predictions, 
        references, 
        model_type=model_type,
        verbose=False
    )
    
    return (
        P.mean().item(),
        R.mean().item(), 
        F1.mean().item()
    )


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match accuracy."""
    matches = sum(1 for p, r in zip(predictions, references) 
                  if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) if predictions else 0.0


def compute_semantic_concept_overlap(
    predicted_concepts: List[List[str]],
    reference_concepts: List[List[str]],
    threshold: float = 0.8
) -> float:
    """
    Compute semantic overlap between concept lists using BERTScore.
    
    For each example, check how many predicted concepts semantically
    match reference concepts (BERTScore > threshold).
    """
    overlaps = []
    
    for pred_list, ref_list in zip(predicted_concepts, reference_concepts):
        if not pred_list or not ref_list:
            overlaps.append(0.0)
            continue
        
        # Compare each predicted concept to all reference concepts
        matched = 0
        for pred in pred_list:
            for ref in ref_list:
                # Quick BERTScore for single pair
                _, _, f1 = bert_score([pred], [ref], 
                                       model_type="microsoft/deberta-base-mnli",
                                       verbose=False)
                if f1.item() > threshold:
                    matched += 1
                    break
        
        overlap = matched / len(pred_list) if pred_list else 0.0
        overlaps.append(overlap)
    
    return np.mean(overlaps)


def evaluate_traces(
    predictions: List[str],
    references: List[str],
    pred_concepts: List[List[str]] = None,
    ref_concepts: List[List[str]] = None,
) -> EvalResult:
    """
    Full evaluation of reasoning trace predictions.
    
    Args:
        predictions: Predicted trace strings
        references: Ground truth trace strings
        pred_concepts: Optional list of predicted concept lists
        ref_concepts: Optional list of reference concept lists
    
    Returns:
        EvalResult with all metrics
    """
    # BERTScore on full traces
    bert_p, bert_r, bert_f1 = compute_bertscore(predictions, references)
    
    # Exact match
    exact = compute_exact_match(predictions, references)
    
    # Semantic concept overlap (if concepts provided)
    sem_overlap = 0.0
    if pred_concepts and ref_concepts:
        sem_overlap = compute_semantic_concept_overlap(pred_concepts, ref_concepts)
    
    return EvalResult(
        bert_precision=bert_p,
        bert_recall=bert_r,
        bert_f1=bert_f1,
        exact_match=exact,
        semantic_overlap=sem_overlap
    )


# ============================================================================
# MEAN ABLATION (replacing zero ablation)
# ============================================================================

class MeanAblation:
    """
    Mean ablation for causal analysis.
    
    Instead of zeroing activations (which creates OOD inputs),
    replace with the mean activation across a reference distribution.
    """
    
    def __init__(self, model, reference_prompts: List[str], device: str = "mps"):
        """
        Initialize with reference prompts to compute mean activations.
        
        Args:
            model: HookedTransformer model
            reference_prompts: List of prompts to compute mean activations from
            device: Device to use
        """
        self.model = model
        self.device = device
        self.n_layers = model.cfg.n_layers
        self.mean_activations = self._compute_mean_activations(reference_prompts)
    
    def _compute_mean_activations(self, prompts: List[str]) -> Dict[int, torch.Tensor]:
        """Compute mean activation at each layer across prompts."""
        print(f"Computing mean activations from {len(prompts)} prompts...")
        
        layer_activations = {i: [] for i in range(self.n_layers)}
        
        with torch.no_grad():
            for prompt in prompts[:100]:  # Limit for efficiency
                _, cache = self.model.run_with_cache(prompt)
                
                for layer in range(self.n_layers):
                    resid = cache[f"blocks.{layer}.hook_resid_post"]
                    # Get last token position activation
                    layer_activations[layer].append(resid[0, -1, :].cpu())
        
        # Compute means
        means = {}
        for layer in range(self.n_layers):
            stacked = torch.stack(layer_activations[layer])
            means[layer] = stacked.mean(dim=0).to(self.device)
        
        return means
    
    def measure_causal_effect(
        self,
        prompt: str,
        layer: int,
    ) -> float:
        """
        Measure causal effect using mean ablation at a specific layer.
        
        Returns: Change in top-1 probability when layer is mean-ablated
        """
        # Get clean output
        clean_logits = self.model(prompt)
        clean_probs = torch.softmax(clean_logits[0, -1], dim=-1)
        clean_top_prob = clean_probs.max().item()
        clean_top_token = clean_probs.argmax().item()
        
        # Define mean ablation hook
        mean_act = self.mean_activations[layer]
        
        def mean_ablation_hook(resid, hook):
            resid[0, -1, :] = mean_act
            return resid
        
        # Run with mean ablation
        ablated_logits = self.model.run_with_hooks(
            prompt,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", mean_ablation_hook)]
        )
        ablated_probs = torch.softmax(ablated_logits[0, -1], dim=-1)
        ablated_prob_of_clean = ablated_probs[clean_top_token].item()
        
        # Causal effect = drop in probability
        causal_effect = clean_top_prob - ablated_prob_of_clean
        
        return max(0, causal_effect)


# ============================================================================
# LOGIT DIFFERENCE METRIC
# ============================================================================

def compute_logit_difference(
    model,
    prompt: str,
    correct_token: str,
    incorrect_token: str,
) -> float:
    """
    Compute logit difference between correct and incorrect tokens.
    
    This is the recommended metric for mechanistic interpretability
    (Zhang & Nanda, ICLR 2024).
    
    Returns: logit(correct) - logit(incorrect)
    """
    logits = model(prompt)
    final_logits = logits[0, -1]
    
    # Get token IDs
    correct_id = model.tokenizer.encode(correct_token, add_special_tokens=False)[0]
    incorrect_id = model.tokenizer.encode(incorrect_token, add_special_tokens=False)[0]
    
    logit_diff = final_logits[correct_id].item() - final_logits[incorrect_id].item()
    
    return logit_diff


def demo_evaluation():
    """Demo the evaluation metrics."""
    
    # Example predictions and references
    predictions = [
        "Texas → Austin",
        "Steve → Jobs",
        "California → Sacramento",
    ]
    
    references = [
        "Dallas → Texas → Austin",
        "Apple → Steve Jobs",
        "Los Angeles → California → Sacramento",
    ]
    
    print("="*60)
    print("Evaluation v2 Demo")
    print("="*60)
    
    print("\nPredictions:")
    for p in predictions:
        print(f"  {p}")
    
    print("\nReferences:")
    for r in references:
        print(f"  {r}")
    
    print("\nComputing BERTScore...")
    result = evaluate_traces(predictions, references)
    
    print(f"\nResults:")
    print(f"  BERTScore Precision: {result.bert_precision:.3f}")
    print(f"  BERTScore Recall:    {result.bert_recall:.3f}")
    print(f"  BERTScore F1:        {result.bert_f1:.3f}")
    print(f"  Exact Match:         {result.exact_match:.3f}")


if __name__ == "__main__":
    demo_evaluation()
