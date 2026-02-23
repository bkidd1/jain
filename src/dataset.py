"""
Dataset Module

Generates and manages datasets for reasoning trace prediction.
Includes task generators for the three core domains:
1. Factual multi-hop reasoning
2. Arithmetic with intermediate steps  
3. Sentiment-influenced generation

NOTE: Prompts are formatted as COMPLETIONS for base models, not questions.
"""

import json
from pathlib import Path
from typing import List, Dict, Generator
from dataclasses import dataclass, asdict


@dataclass
class TaskExample:
    """A single task example with expected reasoning trace."""
    task_type: str
    input_text: str
    expected_output: str
    expected_trace: List[str]  # List of intermediate concepts
    metadata: Dict = None


class MultiHopGenerator:
    """
    Generates factual multi-hop reasoning examples.
    
    Example: "The capital of the state where Dallas is located is"
    Expected trace: Dallas → Texas → Austin
    """
    
    # Knowledge base for generating examples
    CITY_TO_STATE = {
        "Dallas": "Texas", "Houston": "Texas", "Austin": "Texas",
        "Los Angeles": "California", "San Francisco": "California",
        "Seattle": "Washington", "Portland": "Oregon",
        "Miami": "Florida", "Chicago": "Illinois",
        "Boston": "Massachusetts", "Denver": "Colorado",
        "Phoenix": "Arizona", "Atlanta": "Georgia",
    }
    
    STATE_TO_CAPITAL = {
        "Texas": "Austin", "California": "Sacramento",
        "Washington": "Olympia", "Oregon": "Salem",
        "Florida": "Tallahassee", "Illinois": "Springfield",
        "Massachusetts": "Boston", "Colorado": "Denver",
        "Arizona": "Phoenix", "Georgia": "Atlanta",
    }
    
    STATE_TO_LARGEST_CITY = {
        "Texas": "Houston", "California": "Los Angeles",
        "Washington": "Seattle", "Oregon": "Portland",
        "Florida": "Jacksonville", "Illinois": "Chicago",
    }
    
    def generate(self, n: int = 100) -> Generator[TaskExample, None, None]:
        """Generate multi-hop reasoning examples."""
        
        # Type 1: City → State → Capital (completion format)
        for city, state in self.CITY_TO_STATE.items():
            if state in self.STATE_TO_CAPITAL:
                capital = self.STATE_TO_CAPITAL[state]
                yield TaskExample(
                    task_type="multi_hop_capital",
                    input_text=f"The capital of the state where {city} is located is",
                    expected_output=capital,
                    expected_trace=[city, state, capital],
                    metadata={"hop_type": "city_state_capital"}
                )
        
        # Type 2: Simple city → state (for sanity checking)
        for city, state in self.CITY_TO_STATE.items():
            yield TaskExample(
                task_type="city_to_state",
                input_text=f"{city} is a city in the state of",
                expected_output=state,
                expected_trace=[city, state],
                metadata={"hop_type": "city_state_direct"}
            )
        
        # Type 3: Capital → State → Largest City (adversarial)
        for state, capital in self.STATE_TO_CAPITAL.items():
            if state in self.STATE_TO_LARGEST_CITY:
                largest = self.STATE_TO_LARGEST_CITY[state]
                yield TaskExample(
                    task_type="multi_hop_adversarial",
                    input_text=f"The largest city in the state whose capital is {capital} is",
                    expected_output=largest,
                    expected_trace=[capital, state, largest],
                    metadata={"hop_type": "capital_state_largest", "adversarial": True}
                )


class ArithmeticGenerator:
    """
    Generates arithmetic examples with intermediate steps.
    Uses completion format: "37 + 48 ="
    
    Expected trace: 30+40=70, 7+8=15, 70+15=85
    """
    
    def generate(self, n: int = 100) -> Generator[TaskExample, None, None]:
        """Generate arithmetic examples."""
        import random
        
        # Addition (simpler, models are better at this)
        for _ in range(n // 2):
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            result = a + b
            
            yield TaskExample(
                task_type="arithmetic_add",
                input_text=f"{a} + {b} =",
                expected_output=str(result),
                expected_trace=[str(a), str(b), str(result)],
                metadata={"operation": "add", "a": a, "b": b}
            )
        
        # Subtraction
        for _ in range(n // 4):
            a = random.randint(50, 150)
            b = random.randint(10, a - 1)
            result = a - b
            
            yield TaskExample(
                task_type="arithmetic_sub",
                input_text=f"{a} - {b} =",
                expected_output=str(result),
                expected_trace=[str(a), str(b), str(result)],
                metadata={"operation": "subtract", "a": a, "b": b}
            )
        
        # Simple multiplication (single digit × double digit)
        for _ in range(n // 4):
            a = random.randint(2, 9)
            b = random.randint(11, 25)
            result = a * b
            
            yield TaskExample(
                task_type="arithmetic_mult",
                input_text=f"{a} × {b} =",
                expected_output=str(result),
                expected_trace=[str(a), str(b), str(result)],
                metadata={"operation": "multiply", "a": a, "b": b}
            )


class FactualGenerator:
    """
    Generates simple factual completion examples.
    Good for validating logit lens captures known facts.
    """
    
    FACTS = [
        ("The capital of France is", "Paris", ["France", "Paris"]),
        ("The capital of Japan is", "Tokyo", ["Japan", "Tokyo"]),
        ("The capital of Germany is", "Berlin", ["Germany", "Berlin"]),
        ("The CEO of Apple is", "Tim", ["Apple", "Tim"]),
        ("The company that makes Windows is", "Microsoft", ["Windows", "Microsoft"]),
        ("Apple was founded by Steve", "Jobs", ["Apple", "Steve", "Jobs"]),
        ("The author of Harry Potter is", "J", ["Harry Potter", "Rowling"]),
        ("The largest planet in our solar system is", "Jupiter", ["solar system", "Jupiter"]),
        ("Water freezes at", "0", ["freezes", "0"]),
        ("The speed of light is approximately 300,000", "km", ["light", "300"]),
    ]
    
    def generate(self, n: int = 100) -> Generator[TaskExample, None, None]:
        """Generate factual examples."""
        for prompt, expected, trace in self.FACTS:
            yield TaskExample(
                task_type="factual",
                input_text=prompt,
                expected_output=expected,
                expected_trace=trace,
                metadata={"type": "factual_completion"}
            )


class SentimentGenerator:
    """
    Generates sentiment-influenced examples.
    
    Tests whether models are influenced by framing they don't acknowledge.
    Based on Anthropic's "hint" paradigm from the faithfulness study.
    """
    
    NEUTRAL_PROMPTS = [
        ("This movie was", ["good", "bad"]),
        ("The restaurant was", ["great", "terrible"]),
        ("I thought the book was", ["excellent", "boring"]),
    ]
    
    POSITIVE_FRAMES = [
        "Everyone loved it.",
        "Critics praised it highly.",
        "It won many awards.",
    ]
    
    NEGATIVE_FRAMES = [
        "Everyone hated it.",
        "Critics panned it completely.",
        "It was a huge flop.",
    ]
    
    def generate(self, n: int = 100) -> Generator[TaskExample, None, None]:
        """Generate sentiment-influenced examples."""
        
        for prompt, expected_pair in self.NEUTRAL_PROMPTS:
            # Positive framing
            for frame in self.POSITIVE_FRAMES:
                yield TaskExample(
                    task_type="sentiment_positive",
                    input_text=f"{frame} {prompt}",
                    expected_output=expected_pair[0],  # positive word
                    expected_trace=["positive_frame", expected_pair[0]],
                    metadata={"frame_type": "positive", "frame": frame}
                )
            
            # Negative framing
            for frame in self.NEGATIVE_FRAMES:
                yield TaskExample(
                    task_type="sentiment_negative",
                    input_text=f"{frame} {prompt}",
                    expected_output=expected_pair[1],  # negative word
                    expected_trace=["negative_frame", expected_pair[1]],
                    metadata={"frame_type": "negative", "frame": frame}
                )


def generate_full_dataset(output_dir: str = "data/raw") -> Dict[str, int]:
    """Generate the full dataset for all task types."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    counts = {}
    
    # Multi-hop
    multi_hop = list(MultiHopGenerator().generate())
    with open(output_path / "multi_hop.jsonl", "w") as f:
        for example in multi_hop:
            f.write(json.dumps(asdict(example)) + "\n")
    counts["multi_hop"] = len(multi_hop)
    
    # Arithmetic
    arithmetic = list(ArithmeticGenerator().generate(100))
    with open(output_path / "arithmetic.jsonl", "w") as f:
        for example in arithmetic:
            f.write(json.dumps(asdict(example)) + "\n")
    counts["arithmetic"] = len(arithmetic)
    
    # Factual
    factual = list(FactualGenerator().generate())
    with open(output_path / "factual.jsonl", "w") as f:
        for example in factual:
            f.write(json.dumps(asdict(example)) + "\n")
    counts["factual"] = len(factual)
    
    # Sentiment
    sentiment = list(SentimentGenerator().generate())
    with open(output_path / "sentiment.jsonl", "w") as f:
        for example in sentiment:
            f.write(json.dumps(asdict(example)) + "\n")
    counts["sentiment"] = len(sentiment)
    
    print(f"Generated dataset: {counts}")
    return counts


if __name__ == "__main__":
    generate_full_dataset()
