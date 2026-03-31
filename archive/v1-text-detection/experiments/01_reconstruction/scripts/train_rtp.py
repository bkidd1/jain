#!/usr/bin/env python3
"""
Week 3: Train the Reasoning Trace Predictor (RTP).

Fine-tunes a small LM to predict reasoning traces from (prompt, answer) pairs.

Uses LoRA for efficient fine-tuning on limited hardware.
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")


def load_data(train_path: str, test_path: str):
    """Load training and test data."""
    
    def load_jsonl(path):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    
    return train_data, test_data


def format_example(example: dict) -> str:
    """Format a single example for training."""
    return f"{example['input']}\nTrace: {example['target']}"


def tokenize_data(data: list, tokenizer, max_length: int = 128):
    """Tokenize the data."""
    
    texts = [format_example(ex) for ex in data]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["labels"]
    })


def train_rtp(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "models/rtp_v1",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
):
    """Train the RTP model using LoRA."""
    
    print("="*60)
    print("Week 3: Training Reasoning Trace Predictor")
    print("="*60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Model: {model_name}")
    
    # Load data
    print("\nLoading data...")
    train_data, test_data = load_data(
        "data/training/train.jsonl",
        "data/training/test.jsonl"
    )
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Load tokenizer and model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Low rank
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Target attention layers
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize data
    print("\nTokenizing...")
    train_dataset = tokenize_data(train_data, tokenizer)
    test_dataset = tokenize_data(test_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb etc.
        fp16=False,  # MPS doesn't support fp16 training well
        bf16=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Train!
    print(f"\n{'='*60}")
    print("Training...")
    print("="*60)
    
    trainer.train()
    
    # Save
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Test generation
    print(f"\n{'='*60}")
    print("Testing generation...")
    print("="*60)
    
    test_prompts = [
        "Prompt: Dallas is a city in the state of\nAnswer: Texas\nTrace:",
        "Prompt: The capital of France is\nAnswer: Paris\nTrace:",
        "Prompt: Apple was founded by Steve\nAnswer: Jobs\nTrace:",
    ]
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the trace part
        trace = generated.split("Trace:")[-1].strip().split("\n")[0]
        print(f"\n{prompt}")
        print(f"Generated: {trace}")
    
    print(f"\n✓ Training complete! Model saved to {output_dir}")
    
    return model, tokenizer


if __name__ == "__main__":
    train_rtp()
