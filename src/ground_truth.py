"""
Ground Truth Extraction Module

Extracts implicit reasoning traces from language models using:
- Logit lens (intermediate token predictions)
- Linear probes (concept activations)
- Activation patching (causal verification)
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""
    concept: str
    layer: int
    confidence: float
    causally_verified: bool = False


@dataclass
class ReasoningTrace:
    """Full reasoning trace for an (input, output) pair."""
    input_text: str
    output_text: str
    steps: List[ReasoningStep]
    
    def to_string(self) -> str:
        """Convert trace to string format for training."""
        return " → ".join([f"{s.concept}" for s in self.steps if s.causally_verified])


class GroundTruthExtractor:
    """
    Extracts ground truth reasoning traces using mechanistic interpretability.
    
    Pipeline:
    1. Forward pass with hooks to capture all layer activations
    2. Apply logit lens at each layer to get intermediate predictions
    3. Apply linear probes to extract concept activations
    4. Use activation patching to verify causal relevance
    5. Return ordered sequence of causally-verified reasoning steps
    """
    
    def __init__(self, model, tokenizer, device: str = "mps"):
        """
        Initialize the extractor.
        
        Args:
            model: A HookedTransformer from TransformerLens
            tokenizer: The model's tokenizer
            device: Device to run on (mps for Apple Silicon, cuda for NVIDIA)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def get_logit_lens_predictions(
        self, 
        input_text: str, 
        top_k: int = 5
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Apply logit lens at each layer to get intermediate token predictions.
        
        The logit lens projects each layer's residual stream through the 
        unembedding matrix to see what tokens the model "would predict"
        at that layer.
        
        Args:
            input_text: The input prompt
            top_k: Number of top predictions to return per layer
            
        Returns:
            Dict mapping layer index to list of (token, probability) tuples
        """
        # TODO: Implement with TransformerLens hooks
        # Example structure:
        # {
        #     0: [("the", 0.3), ("a", 0.2), ...],
        #     1: [("Texas", 0.4), ("state", 0.2), ...],
        #     ...
        # }
        raise NotImplementedError("Implement with TransformerLens")
    
    def apply_activation_patching(
        self,
        input_text: str,
        candidate_concept: str,
        layer: int
    ) -> float:
        """
        Verify causal relevance of a concept using activation patching.
        
        Ablates the candidate concept's representation and measures
        the change in output. Large changes indicate the concept was
        actually *used* in reasoning, not just present.
        
        Args:
            input_text: The input prompt
            candidate_concept: The concept to test
            layer: The layer where the concept was detected
            
        Returns:
            Causal importance score (higher = more causally relevant)
        """
        # TODO: Implement activation patching
        # This is CRITICAL - without this step, we're just finding
        # information that's present, not information that's used
        raise NotImplementedError("Implement activation patching")
    
    def extract_trace(
        self,
        input_text: str,
        output_text: str,
        causal_threshold: float = 0.1
    ) -> ReasoningTrace:
        """
        Extract full reasoning trace for an (input, output) pair.
        
        Args:
            input_text: The input prompt
            output_text: The model's output
            causal_threshold: Minimum causal importance to include a step
            
        Returns:
            ReasoningTrace with causally-verified reasoning steps
        """
        # 1. Get logit lens predictions at each layer
        layer_predictions = self.get_logit_lens_predictions(input_text)
        
        # 2. Filter to candidate concepts (tokens that appear in predictions
        #    but not in input, suggesting intermediate reasoning)
        candidates = self._find_candidate_concepts(input_text, layer_predictions)
        
        # 3. Verify each candidate with activation patching
        verified_steps = []
        for concept, layer, confidence in candidates:
            causal_score = self.apply_activation_patching(
                input_text, concept, layer
            )
            if causal_score >= causal_threshold:
                verified_steps.append(ReasoningStep(
                    concept=concept,
                    layer=layer,
                    confidence=confidence,
                    causally_verified=True
                ))
        
        # 4. Sort by layer to get ordered trace
        verified_steps.sort(key=lambda s: s.layer)
        
        return ReasoningTrace(
            input_text=input_text,
            output_text=output_text,
            steps=verified_steps
        )
    
    def _find_candidate_concepts(
        self,
        input_text: str,
        layer_predictions: Dict[int, List[Tuple[str, float]]]
    ) -> List[Tuple[str, int, float]]:
        """Find concepts that appear in predictions but not input."""
        input_tokens = set(input_text.lower().split())
        candidates = []
        
        for layer, predictions in layer_predictions.items():
            for token, confidence in predictions:
                if token.lower() not in input_tokens:
                    candidates.append((token, layer, confidence))
        
        return candidates


# Convenience function for quick testing
def demo_logit_lens():
    """Demo the logit lens on a simple example."""
    try:
        from transformer_lens import HookedTransformer
        
        print("Loading model...")
        model = HookedTransformer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            device="mps"  # Apple Silicon
        )
        
        prompt = "The capital of Texas is"
        print(f"\nPrompt: {prompt}")
        print("\nLogit lens predictions by layer:")
        
        # Run with cache to get all activations
        _, cache = model.run_with_cache(prompt)
        
        # Apply logit lens at each layer
        for layer in range(0, model.cfg.n_layers, 4):  # Every 4th layer
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            logits = model.unembed(model.ln_final(resid))
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_tokens = torch.topk(probs, 5)
            
            print(f"\nLayer {layer}:")
            for prob, idx in zip(top_tokens.values, top_tokens.indices):
                token = model.tokenizer.decode([idx])
                print(f"  {token!r}: {prob:.3f}")
                
    except Exception as e:
        print(f"Demo requires model download. Error: {e}")


if __name__ == "__main__":
    demo_logit_lens()
