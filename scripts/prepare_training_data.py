#!/usr/bin/env python3
"""
Week 3: Prepare training data for RTP.

Formats extracted traces into (input, target) pairs for fine-tuning.
Format: 
  Input: "Prompt: {prompt}\nAnswer: {answer}"
  Target: "{trace_string}"
"""

import json
from pathlib import Path
import random


def load_traces(path: str = "data/processed/traces_llama.jsonl"):
    """Load extracted traces."""
    traces = []
    with open(path) as f:
        for line in f:
            traces.append(json.loads(line))
    return traces


def format_for_training(traces: list) -> list:
    """Format traces for RTP training."""
    
    formatted = []
    
    for t in traces:
        input_text = t["input_text"]
        extracted = t["extracted"]
        final_output = extracted["final_output"]
        trace_string = extracted["trace_string"]
        
        # Skip if trace is empty or just the output
        if not trace_string or trace_string == final_output:
            # Use the final output as a minimal trace
            trace_string = final_output
        
        # Skip garbage traces (single char, punctuation only)
        if len(trace_string) < 2 or trace_string in ["?", ".", ",", "!"]:
            continue
        
        # Format as training example
        example = {
            "task_type": t["task_type"],
            "input": f"Prompt: {input_text}\nAnswer: {final_output}",
            "target": trace_string,
            "metadata": {
                "original_prompt": input_text,
                "final_output": final_output,
                "concepts": extracted.get("concepts", [])
            }
        }
        formatted.append(example)
    
    return formatted


def split_data(data: list, train_ratio: float = 0.8):
    """Split into train/test sets, stratified by task type."""
    
    # Group by task type
    by_type = {}
    for ex in data:
        t = ex["task_type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(ex)
    
    train, test = [], []
    
    for task_type, examples in by_type.items():
        random.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        train.extend(examples[:split_idx])
        test.extend(examples[split_idx:])
    
    random.shuffle(train)
    random.shuffle(test)
    
    return train, test


def main():
    print("="*60)
    print("Week 3: Preparing RTP training data")
    print("="*60)
    
    # Load traces
    traces = load_traces()
    print(f"\nLoaded {len(traces)} traces")
    
    # Format for training
    formatted = format_for_training(traces)
    print(f"Formatted {len(formatted)} examples (filtered out empty/garbage)")
    
    # Show distribution
    by_type = {}
    for ex in formatted:
        t = ex["task_type"]
        by_type[t] = by_type.get(t, 0) + 1
    
    print(f"\nBy task type:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")
    
    # Split
    train, test = split_data(formatted)
    print(f"\nSplit: {len(train)} train, {len(test)} test")
    
    # Save
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    
    with open(output_dir / "test.jsonl", "w") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\n✓ Saved to {output_dir}/")
    
    # Show examples
    print(f"\n{'='*60}")
    print("Sample training examples")
    print("="*60)
    
    for ex in train[:5]:
        print(f"\n[{ex['task_type']}]")
        print(f"Input: {ex['input']}")
        print(f"Target: {ex['target']}")


if __name__ == "__main__":
    random.seed(42)
    main()
