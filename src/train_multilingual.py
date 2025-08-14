#!/usr/bin/env python3
"""
Multilingual Sarcasm Detection Training Script
Supports training on English and Hindi datasets with language-specific models
"""

import os
import argparse
import yaml
from typing import Dict, List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset as HFDataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

from src.data.datasets import MultilingualSarcasmDataset, MultilingualDataCollator
from src.models.sarcasm_model import MultilingualSarcasmClassifier, create_multilingual_model
from src.utils import set_seed, ensure_special_tokens, create_language_tokenizer
from src.commonsense.conceptnet import synthesize_hint

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual Sarcasm Detection Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--language", type=str, choices=["en", "hi", "both"], default="both", 
                       help="Target language for training")
    parser.add_argument("--output_dir", type=str, default="outputs/multilingual", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def compute_multilingual_metrics(eval_pred):
    """Compute metrics for multilingual evaluation"""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    
    # Overall metrics
    macro_f1 = f1_score(labels, preds, average="macro")
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    
    try:
        probs = torch.tensor(logits).softmax(-1).numpy()[:,1]
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = float("nan")
    
    return {
        "macro_f1": macro_f1, 
        "f1_sarcastic": f1, 
        "precision": p, 
        "recall": r, 
        "auroc": auroc
    }

class MultilingualSarcasmTrainer(Trainer):
    """Custom trainer for multilingual sarcasm detection"""
    
    def __init__(self, lambda_contrastive: float = 0.2, use_contrastive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.use_contrastive = use_contrastive
        self.lambda_contrastive = lambda_contrastive

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        pair_idx = inputs.pop("pair_idx", None)
        language_ids = inputs.pop("language_ids", None)
        
        # Convert language strings to IDs
        if language_ids and isinstance(language_ids[0], str):
            lang_to_id = {"en": 0, "hi": 1}
            language_ids = torch.tensor([lang_to_id.get(lang, 0) for lang in language_ids])
            inputs["language_ids"] = language_ids
        
        outputs = model(**inputs)
        logits = outputs["logits"]
        
        # Classification loss
        loss_fct = nn.CrossEntropyLoss()
        ce_loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        total_loss = ce_loss
        
        # Contrastive loss if enabled
        if self.use_contrastive and pair_idx is not None:
            proj = outputs["proj"]
            from src.models.sarcasm_model import info_nce_loss
            c_loss = info_nce_loss(proj, pair_idx, temperature=0.07)
            total_loss = ce_loss + self.lambda_contrastive * c_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

def create_language_datasets(train_path: str, val_path: str, language: str, config: Dict):
    """Create language-specific datasets"""
    if language == "both":
        # Load both languages
        train_en = MultilingualSarcasmDataset(train_path, target_language="en")
        train_hi = MultilingualSarcasmDataset(train_path, target_language="hi")
        val_en = MultilingualSarcasmDataset(val_path, target_language="en")
        val_hi = MultilingualSarcasmDataset(val_path, target_language="hi")
        
        # Combine datasets
        train_combined = train_en.items + train_hi.items
        val_combined = val_en.items + val_hi.items
        
        return train_combined, val_combined
    else:
        # Load specific language
        train_ds = MultilingualSarcasmDataset(train_path, target_language=language)
        val_ds = MultilingualSarcasmDataset(val_path, target_language=language)
        return train_ds.items, val_ds.items

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    train_items, val_items = create_language_datasets(
        args.train_path, args.val_path, args.language, config
    )
    
    print(f"Training examples: {len(train_items)}")
    print(f"Validation examples: {len(val_items)}")
    
    # Language distribution
    lang_counts = {}
    for item in train_items:
        lang = item.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"Language distribution: {lang_counts}")
    
    # Create tokenizer (use English tokenizer for now, will be updated per language)
    base_model = config.get("model_name", "roberta-base")
    tokenizer = create_language_tokenizer(base_model, "en")
    ensure_special_tokens(tokenizer)
    
    # Create data collator
    lang_config = config.get("languages", {}).get("en", config)
    collator = MultilingualDataCollator(
        tokenizer=tokenizer,
        max_length=lang_config.get("max_length", 192),
        commonsense_fn=(synthesize_hint if lang_config.get("use_commonsense", False) else None),
        prepend_commonsense=lang_config.get("use_commonsense", False)
    )
    
    # Convert to HuggingFace datasets
    hf_train = HFDataset.from_list(train_items)
    hf_val = HFDataset.from_list(val_items)
    
    # Create model
    if args.language == "both":
        # Use ensemble model for multilingual
        from src.models.sarcasm_model import EnsembleMultilingualClassifier
        model = EnsembleMultilingualClassifier(
            english_model_name=config["languages"]["en"]["model_name"],
            hindi_model_name=config["languages"]["hi"]["model_name"],
            num_labels=2
        )
    else:
        # Use language-specific model
        model = create_multilingual_model(config, args.language)
    
    # Resize token embeddings if needed
    if hasattr(model, 'encoder'):
        model.encoder.resize_token_embeddings(len(tokenizer))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=max(8, config.get("batch_size", 16)),
        num_train_epochs=config.get("epochs", 3),
        learning_rate=config.get("lr", 2e-5),
        warmup_steps=config.get("training", {}).get("warmup_steps", 500),
        weight_decay=config.get("training", {}).get("weight_decay", 0.01),
        gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 1),
        fp16=config.get("training", {}).get("fp16", True),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
        save_total_limit=3,
    )
    
    # Create trainer
    trainer = MultilingualSarcasmTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_multilingual_metrics,
        use_contrastive=lang_config.get("use_contrastive", False),
        lambda_contrastive=lang_config.get("lambda_contrastive", 0.2)
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save config
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    print(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
