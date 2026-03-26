#!/usr/bin/env python3
"""Quick test: does excluding target architecture help with matched prompts?"""

import json
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from torch.utils.data import Dataset

class ResponseDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Response-only: just the model's response
        text = ex["response"]
        label = 1 if ex["label"] == "unfaithful" else 0
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)
    
    try:
        auroc = roc_auc_score(labels, probs)
    except:
        auroc = 0.5
    
    return {
        "auroc": auroc,
        "accuracy": accuracy_score(labels, preds),
    }

def train_and_eval(train_data, test_data, name, output_dir):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Train: {len(train_data)} examples")
    print(f"Test: {len(test_data)} examples")
    
    train_unfaithful = sum(1 for ex in train_data if ex["label"] == "unfaithful")
    test_unfaithful = sum(1 for ex in test_data if ex["label"] == "unfaithful")
    print(f"Train unfaithful: {train_unfaithful} ({100*train_unfaithful/len(train_data):.1f}%)")
    print(f"Test unfaithful: {test_unfaithful} ({100*test_unfaithful/len(test_data):.1f}%)")
    print(f"{'='*60}")
    
    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float32
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    # Datasets
    train_dataset = ResponseDataset(train_data, tokenizer)
    test_dataset = ResponseDataset(test_data, tokenizer)
    
    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for speed
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Final eval
    results = trainer.evaluate()
    print(f"\n*** {name} AUROC: {results['eval_auroc']:.4f} ***\n")
    
    return results['eval_auroc']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/extractions")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load data
    all_data = load_jsonl(data_dir / "combined_v2.jsonl")
    tinyllama_data = [ex for ex in all_data if "TinyLlama" in ex.get("source_model", "")]
    non_tinyllama_data = [ex for ex in all_data if "TinyLlama" not in ex.get("source_model", "")]
    
    print(f"Total: {len(all_data)}")
    print(f"TinyLlama: {len(tinyllama_data)}")
    print(f"Non-TinyLlama (Qwen+Phi2): {len(non_tinyllama_data)}")
    
    # Test 1: Train on Qwen+Phi2 only, eval on TinyLlama
    auroc_exclude = train_and_eval(
        non_tinyllama_data, 
        tinyllama_data,
        "EXCLUDE TinyLlama (train Qwen+Phi2)",
        "/tmp/exclude_target"
    )
    
    # Test 2: Train on all, eval on TinyLlama  
    auroc_include = train_and_eval(
        all_data,
        tinyllama_data,
        "INCLUDE TinyLlama (train all)",
        "/tmp/include_target"
    )
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Exclude TinyLlama from training: {auroc_exclude:.4f}")
    print(f"Include TinyLlama in training:   {auroc_include:.4f}")
    print(f"Difference (exclude - include):  {auroc_exclude - auroc_include:+.4f}")
    print("="*60)
    
    if auroc_exclude > auroc_include + 0.05:
        print("→ Effect PERSISTS with matched prompts (likely real)")
    elif auroc_include > auroc_exclude + 0.05:
        print("→ Effect REVERSES (was an artifact)")
    else:
        print("→ No significant difference (effect disappears)")

if __name__ == "__main__":
    main()
