#!/usr/bin/env python3
"""
Train divergence detector with configurable input format.

Supports:
- full: prompt + response (baseline, replicates exp 02)
- response-only: just the generated CoT
- redacted: question without hint preamble + response
"""

import json
import torch
import argparse
import re
from pathlib import Path
from typing import Literal
import random

from torch.utils.data import Dataset
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


InputFormat = Literal["full", "response-only", "redacted"]


def extract_question(prompt: str) -> str:
    """Extract just the question, removing hint preambles like 'I recall that X might be the answer.'"""
    # Pattern: "I recall that ... might be the answer. <question>"
    # or similar hint preambles
    patterns = [
        r"I recall that .+? might be the answer\.\s*",
        r"I think the answer is .+?\.\s*",
        r"Someone told me .+?\.\s*",
    ]
    result = prompt
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result.strip()


def format_input(example: dict, input_format: InputFormat) -> str:
    """Format input based on ablation condition."""
    if input_format == "full":
        return f"{example['prompt']}\n\nResponse: {example['generated_text']}"
    elif input_format == "response-only":
        return example['generated_text'].strip()
    elif input_format == "redacted":
        question = extract_question(example['prompt'])
        return f"{question}\n\nResponse: {example['generated_text']}"
    else:
        raise ValueError(f"Unknown input format: {input_format}")


class AblationDataset(Dataset):
    """Dataset for ablation study."""
    
    def __init__(self, examples: list[dict], tokenizer, input_format: InputFormat, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.input_format = input_format
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = format_input(ex, self.input_format)
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
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
    
    random.seed(42)
    random.shuffle(examples)
    
    split_idx = int(len(examples) * (1 - val_split))
    return examples[:split_idx], examples[split_idx:]


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs > 0.5).astype(int)
    
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = 0.5
    
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
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--input-format", type=str, choices=["full", "response-only", "redacted"], 
                        default="response-only", help="Input format for ablation")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input format: {args.input_format}")
    print(f"Loading data from {args.data}")
    
    train_examples, val_examples = load_data(Path(args.data))
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")
    
    # Show example of input format
    print(f"\nExample input ({args.input_format}):")
    print("-" * 40)
    print(format_input(train_examples[0], args.input_format)[:300])
    print("-" * 40)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        torch_dtype=torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Datasets
    train_dataset = AblationDataset(train_examples, tokenizer, args.input_format)
    val_dataset = AblationDataset(val_examples, tokenizer, args.input_format)
    
    # Training
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
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nFinal evaluation:")
    results = trainer.evaluate()
    for key, value in sorted(results.items()):
        print(f"  {key}: {value:.4f}")
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    config = {
        "input_format": args.input_format,
        "model_name": args.model,
        "lora_r": args.lora_r,
        "epochs": args.epochs,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "final_metrics": {k: float(v) for k, v in results.items()},
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
