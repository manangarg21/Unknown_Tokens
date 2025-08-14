#!/usr/bin/env python3
"""
Multilingual Sarcasm Detection Evaluation Script
Supports evaluation of English and Hindi models with language-specific analysis
"""

import argparse
import yaml
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import os

from src.data.datasets import MultilingualSarcasmDataset, MultilingualDataCollator
from src.models.sarcasm_model import MultilingualSarcasmClassifier, create_multilingual_model
from src.utils import ensure_special_tokens, detect_language
from src.commonsense.conceptnet import synthesize_hint

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual Sarcasm Detection Evaluation")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--language", type=str, choices=["en", "hi", "both"], default="both", 
                       help="Target language for evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results")
    parser.add_argument("--max_length", type=int, default=192, help="Maximum sequence length")
    parser.add_argument("--use_commonsense", action="store_true", help="Use commonsense hints")
    return parser.parse_args()

@torch.no_grad()
def evaluate_model(model, tokenizer, dataset, collator, device, language):
    """Evaluate model on dataset"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_texts = []
    all_languages = []
    
    # Process in batches
    batch_size = 8
    for i in range(0, len(dataset), batch_size):
        batch_items = dataset[i:i+batch_size]
        
        # Prepare batch
        batch_features = []
        for item in batch_items:
            batch_features.append({
                "text": item["text"],
                "label": item["label"],
                "context": item.get("context"),
                "language": item.get("language", language)
            })
        
        # Collate batch
        batch = collator(batch_features)
        
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        # Get predictions
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        # Store results
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_texts.extend([item["text"] for item in batch_items])
        all_languages.extend([item.get("language", language) for item in batch_items])
    
    return {
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probabilities,
        "texts": all_texts,
        "languages": all_languages
    }

def analyze_language_performance(results, language):
    """Analyze performance for specific language"""
    lang_mask = [lang == language for lang in results["languages"]]
    
    if not any(lang_mask):
        return None
    
    lang_predictions = np.array(results["predictions"])[lang_mask]
    lang_labels = np.array(results["labels"])[lang_mask]
    lang_probabilities = np.array(results["probabilities"])[lang_mask]
    
    # Calculate metrics
    f1 = f1_score(lang_labels, lang_predictions, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(
        lang_labels, lang_predictions, average="binary", zero_division=0
    )
    
    try:
        auroc = roc_auc_score(lang_labels, lang_probabilities[:, 1])
    except:
        auroc = float("nan")
    
    return {
        "language": language,
        "f1_macro": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "num_samples": len(lang_labels)
    }

def save_detailed_results(results, output_dir, language):
    """Save detailed evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results DataFrame
    df = pd.DataFrame({
        "text": results["texts"],
        "true_label": results["labels"],
        "predicted_label": results["predictions"],
        "sarcasm_probability": [prob[1] for prob in results["probabilities"]],
        "language": results["languages"],
        "correct": [pred == label for pred, label in zip(results["predictions"], results["labels"])]
    })
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"{language}_evaluation_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(results["labels"], results["predictions"])
    cm_df = pd.DataFrame(cm, columns=["Non-sarcastic", "Sarcastic"], 
                        index=["Non-sarcastic", "Sarcastic"])
    cm_path = os.path.join(output_dir, f"{language}_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = MultilingualSarcasmDataset(args.val_path, target_language=args.language)
    print(f"Loaded {len(dataset)} validation examples")
    
    # Language distribution
    lang_counts = {}
    for item in dataset:
        lang = item.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"Language distribution: {lang_counts}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    ensure_special_tokens(tokenizer)
    
    # Create model
    if args.language == "both":
        # Load ensemble model
        from src.models.sarcasm_model import EnsembleMultilingualClassifier
        model = EnsembleMultilingualClassifier(
            english_model_name=config["languages"]["en"]["model_name"],
            hindi_model_name=config["languages"]["hi"]["model_name"],
            num_labels=2
        )
        # Load state dict
        model.load_state_dict(torch.load(os.path.join(args.model_dir, "pytorch_model.bin")))
    else:
        model = create_multilingual_model(config, args.language)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, "pytorch_model.bin")))
    
    # Create data collator
    lang_config = config.get("languages", {}).get("en", config)
    collator = MultilingualDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        commonsense_fn=(synthesize_hint if args.use_commonsense else None),
        prepend_commonsense=args.use_commonsense
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate_model(model, tokenizer, dataset, collator, device, args.language)
    
    # Overall performance
    print("\n" + "="*50)
    print("OVERALL PERFORMANCE")
    print("="*50)
    
    f1 = f1_score(results["labels"], results["predictions"], average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(
        results["labels"], results["predictions"], average="binary", zero_division=0
    )
    
    try:
        auroc = roc_auc_score(results["labels"], [prob[1] for prob in results["probabilities"]])
    except:
        auroc = float("nan")
    
    print(f"Macro F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUROC: {auroc:.4f}")
    
    # Language-specific performance
    if args.language == "both":
        print("\n" + "="*50)
        print("LANGUAGE-SPECIFIC PERFORMANCE")
        print("="*50)
        
        for lang in ["en", "hi"]:
            lang_results = analyze_language_performance(results, lang)
            if lang_results:
                print(f"\n{lang.upper()}:")
                print(f"  F1 Macro: {lang_results['f1_macro']:.4f}")
                print(f"  Precision: {lang_results['precision']:.4f}")
                print(f"  Recall: {lang_results['recall']:.4f}")
                print(f"  AUROC: {lang_results['auroc']:.4f}")
                print(f"  Samples: {lang_results['num_samples']}")
                
                # Save detailed results for each language
                save_detailed_results(results, args.output_dir, lang)
    
    # Save overall results
    save_detailed_results(results, args.output_dir, "overall")
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
