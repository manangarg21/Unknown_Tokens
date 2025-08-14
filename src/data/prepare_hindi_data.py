#!/usr/bin/env python3
"""
Hindi Sarcasm Dataset Preparation Script
Converts various Hindi sarcasm datasets to the unified format
"""

import json
import argparse
import pandas as pd
import os
from typing import List, Dict
import re

def clean_hindi_text(text: str) -> str:
    """Clean and normalize Hindi text"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions and hashtags (keep the text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    return text

def from_hindi_twitter_csv(csv_path: str, out_csv: str):
    """Convert Hindi Twitter sarcasm dataset to unified format"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} examples from {csv_path}")
        
        rows = []
        for idx, row in df.iterrows():
            # Handle different column names
            text_col = None
            label_col = None
            
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['text', 'tweet', 'comment', 'post']):
                    text_col = col
                if any(keyword in col.lower() for keyword in ['label', 'sarcasm', 'is_sarcastic', 'class']):
                    label_col = col
            
            if text_col is None or label_col is None:
                print(f"Warning: Could not identify text or label columns in {csv_path}")
                print(f"Available columns: {list(df.columns)}")
                return
            
            text = clean_hindi_text(str(row[text_col]))
            if not text:
                continue
                
            # Convert label to binary (assuming 1=sarcastic, 0=non-sarcastic)
            try:
                label = int(row[label_col])
                if label not in [0, 1]:
                    # Try to map other labels
                    if str(row[label_col]).lower() in ['sarcastic', 'true', 'yes', '1']:
                        label = 1
                    else:
                        label = 0
            except:
                # Default to non-sarcastic if conversion fails
                label = 0
            
            rows.append({
                "text": text,
                "label": label,
                "context": "",
                "pair_id": "",
                "language": "hi"
            })
        
        # Save to unified format
        output_df = pd.DataFrame(rows)
        output_df.to_csv(out_csv, index=False)
        print(f"Saved {len(rows)} Hindi examples to {out_csv}")
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

def from_hindi_jsonl(jsonl_path: str, out_csv: str):
    """Convert Hindi JSONL sarcasm dataset to unified format"""
    rows = []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    obj = json.loads(line)
                    
                    # Extract text and label
                    text = obj.get('text', obj.get('tweet', obj.get('comment', '')))
                    text = clean_hindi_text(str(text))
                    
                    if not text:
                        continue
                    
                    # Extract label
                    label = obj.get('label', obj.get('sarcasm', obj.get('is_sarcastic', 0)))
                    try:
                        label = int(label)
                        if label not in [0, 1]:
                            label = 0
                    except:
                        label = 0
                    
                    rows.append({
                        "text": text,
                        "label": label,
                        "context": obj.get('context', ''),
                        "pair_id": obj.get('pair_id', ''),
                        "language": "hi"
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num + 1}: {e}")
                    continue
        
        # Save to unified format
        output_df = pd.DataFrame(rows)
        output_df.to_csv(out_csv, index=False)
        print(f"Saved {len(rows)} Hindi examples to {out_csv}")
        
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")

def create_hindi_synthetic_data(out_csv: str, num_examples: int = 1000):
    """Create synthetic Hindi sarcasm examples for testing"""
    
    # Hindi sarcastic patterns
    sarcastic_patterns = [
        "बहुत अच्छा, {context}",
        "क्या बात है, {context}",
        "वाह, {context}",
        "बिल्कुल सही, {context}",
        "कमाल है, {context}",
        "शानदार, {context}",
        "बेहतरीन, {context}"
    ]
    
    # Hindi non-sarcastic patterns
    non_sarcastic_patterns = [
        "यह {context} है",
        "मुझे {context} पसंद है",
        "{context} अच्छा लगता है",
        "यह {context} बहुत अच्छा है"
    ]
    
    # Context templates
    contexts = [
        "आज का दिन", "यह काम", "यह समस्या", "यह स्थिति", "यह परिणाम",
        "यह अनुभव", "यह समाधान", "यह प्रयास", "यह प्रक्रिया", "यह परिणाम"
    ]
    
    rows = []
    
    # Generate sarcastic examples
    for i in range(num_examples // 2):
        pattern = sarcastic_patterns[i % len(sarcastic_patterns)]
        context = contexts[i % len(contexts)]
        text = pattern.format(context=context)
        
        rows.append({
            "text": text,
            "label": 1,  # Sarcastic
            "context": "",
            "pair_id": f"synth_s_{i}",
            "language": "hi"
        })
    
    # Generate non-sarcastic examples
    for i in range(num_examples // 2):
        pattern = non_sarcastic_patterns[i % len(non_sarcastic_patterns)]
        context = contexts[i % len(contexts)]
        text = pattern.format(context=context)
        
        rows.append({
            "text": text,
            "label": 0,  # Non-sarcastic
            "context": "",
            "pair_id": f"synth_ns_{i}",
            "language": "hi"
        })
    
    # Save to CSV
    output_df = pd.DataFrame(rows)
    output_df.to_csv(out_csv, index=False)
    print(f"Created {len(rows)} synthetic Hindi examples in {out_csv}")

def validate_hindi_dataset(csv_path: str):
    """Validate Hindi dataset and show statistics"""
    try:
        df = pd.read_csv(csv_path)
        print(f"\nDataset Statistics for {csv_path}:")
        print(f"Total examples: {len(df)}")
        print(f"Language distribution: {df['language'].value_counts().to_dict()}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        # Check for Hindi characters
        hindi_count = 0
        for text in df['text']:
            if re.search(r'[\u0900-\u097F]', str(text)):
                hindi_count += 1
        
        print(f"Texts with Hindi characters: {hindi_count}")
        
        # Show some examples
        print("\nSample examples:")
        for i, row in df.head(3).iterrows():
            print(f"  {i+1}. Label: {row['label']}, Text: {row['text'][:50]}...")
            
    except Exception as e:
        print(f"Error validating {csv_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Hindi Sarcasm Datasets")
    parser.add_argument("--task", choices=["twitter", "jsonl", "synthetic", "validate"], required=True)
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of synthetic examples")
    
    args = parser.parse_args()
    
    if args.task == "twitter":
        if not args.input:
            print("Error: --input required for twitter task")
            return
        from_hindi_twitter_csv(args.input, args.output)
        
    elif args.task == "jsonl":
        if not args.input:
            print("Error: --input required for jsonl task")
            return
        from_hindi_jsonl(args.input, args.output)
        
    elif args.task == "synthetic":
        create_hindi_synthetic_data(args.output, args.num_examples)
        
    elif args.task == "validate":
        if not args.input:
            print("Error: --input required for validate task")
            return
        validate_hindi_dataset(args.input)
    
    # Validate output if created
    if args.task != "validate" and os.path.exists(args.output):
        print(f"\nValidating output file:")
        validate_hindi_dataset(args.output)

if __name__ == "__main__":
    main()
