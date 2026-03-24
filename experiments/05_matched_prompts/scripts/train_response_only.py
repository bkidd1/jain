#!/usr/bin/env python3
"""
Train detector on RESPONSE ONLY - no prompt information.

This is the clean test: can we detect hint influence from CoT text alone?
"""

import json
import torch
import argparse
import random
from pathlib import Path

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


class ResponseOnlyDataset(Dataset):
    """Dataset using only the response text, no prompt."""
    
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # RESPONSE ONLY - this is the key
        text = ex['response'].strip()
        
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


def load_data(data_paths: list[Path], val_split: float = 0.15):
    """Load and combine data from multiple extraction files."""
    examples = []
    for path in data_paths:
        with open(path) as f:
            for line in f:
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} total examples from {len(data_paths)} files")
    
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
    parser.add_argument("--data", type=str, nargs="+", required=True,
                        help="Extraction JSONL files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_paths = [Path(p) for p in args.data]
    train_examples, val_examples = load_data(data_paths)
    
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")
    
    # Count labels
    train_unfaithful = sum(1 for ex in train_examples if ex['label'] == 'unfaithful')
    val_unfaithful = sum(1 for ex in val_examples if ex['label'] == 'unfaithful')
    print(f"Train unfaithful: {train_unfaithful} ({100*train_unfaithful/len(train_examples):.1f}%)")
    print(f"Val unfaithful: {val_unfaithful} ({100*val_unfaithful/len(val_examples):.1f}%)")
    
    # Show example
    print(f"\n=== Example response (what the model sees) ===")
    print(train_examples[0]['response'][:300])
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        torch_dtype=torch.float32,  # float32 for stability
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
    train_dataset = ResponseOnlyDataset(train_examples, tokenizer)
    val_dataset = ResponseOnlyDataset(val_examples, tokenizer)
    
    # Training
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
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
    
    print("\n=== Training RESPONSE-ONLY detector ===")
    trainer.train()
    
    print("\nFinal evaluation:")
    results = trainer.evaluate()
    for key, value in sorted(results.items()):
        print(f"  {key}: {value:.4f}")
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    config = {
        "input_format": "response-only",
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
    print(f"\n*** RESPONSE-ONLY AUROC: {results['eval_auroc']:.4f} ***")


if __name__ == "__main__":
    main()
