#!/usr/bin/env python3
"""
Train a divergence detector: binary classifier for faithful vs unfaithful reasoning.

Input: prompt + generated CoT
Output: probability of unfaithful reasoning

Architecture: TinyLlama backbone with classification head (LoRA fine-tuning)
"""

import json
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import random

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np


class DivergenceDataset(Dataset):
    """Dataset for divergence detection."""
    
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Format: "<prompt>\n\nResponse: <generated_text>"
        text = f"{ex['prompt']}\n\nResponse: {ex['generated_text']}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        # Label: 0 = faithful, 1 = unfaithful
        label = 1 if ex['label'] == 'unfaithful' else 0
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': label,
        }


def load_data(data_path: Path, val_split: float = 0.15):
    """Load and split data."""
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Shuffle with seed
    random.seed(42)
    random.shuffle(examples)
    
    # Split
    split_idx = int(len(examples) * (1 - val_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    return train_examples, val_examples


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    
    # Get predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs > 0.5).astype(int)
    
    # Metrics
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = 0.5  # If only one class present
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    accuracy = (preds == labels).mean()
    
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=str, required=True, help="Path to extraction JSONL")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    
    # Paths
    data_path = Path(args.data)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_path.parent.parent / "models" / "detector_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {data_path}")
    train_examples, val_examples = load_data(data_path)
    
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")
    
    # Count labels
    train_unfaithful = sum(1 for ex in train_examples if ex['label'] == 'unfaithful')
    val_unfaithful = sum(1 for ex in val_examples if ex['label'] == 'unfaithful')
    print(f"Train unfaithful: {train_unfaithful} ({100*train_unfaithful/len(train_examples):.1f}%)")
    print(f"Val unfaithful: {val_unfaithful} ({100*val_unfaithful/len(val_examples):.1f}%)")
    
    # Load tokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        torch_dtype=torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create datasets
    train_dataset = DivergenceDataset(train_examples, tokenizer)
    val_dataset = DivergenceDataset(val_examples, tokenizer)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        logging_steps=10,
        fp16=False,  # MPS doesn't support fp16 training
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation:")
    results = trainer.evaluate()
    for key, value in sorted(results.items()):
        print(f"  {key}: {value:.4f}")
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config = {
        "model_name": args.model,
        "lora_r": args.lora_r,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "final_metrics": {k: float(v) for k, v in results.items()},
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
