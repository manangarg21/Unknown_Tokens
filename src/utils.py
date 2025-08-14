import random, numpy as np, torch
from typing import Optional, Dict, List
import re
import langdetect
from langdetect import detect, DetectorFactory

# Set seed for langdetect to ensure consistent results
DetectorFactory.seed = 0

SPECIAL_TOKENS = ["[KNOW]", "[PARENT]", "[TEXT]", "[LANG_EN]", "[LANG_HI]"]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_special_tokens(tokenizer):
    added = tokenizer.add_tokens([t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()])
    return added

def join_with_context(text: str, context: Optional[str]):
    if context and len(str(context).strip()) > 0:
        first = "[PARENT] " + str(context).strip()
        second = "[TEXT] " + str(text).strip()
        return first, second
    else:
        return "[TEXT] " + str(text).strip(), None

def detect_language(text: str) -> str:
    """
    Detect language of text, returning 'en' for English, 'hi' for Hindi, or 'unknown'
    """
    try:
        # Clean text for better detection
        clean_text = re.sub(r'[^\w\s]', '', text)
        if len(clean_text.strip()) < 3:
            return "unknown"
        
        detected = detect(clean_text)
        
        # Map language codes to our supported languages
        if detected in ['en']:
            return 'en'
        elif detected in ['hi', 'mr', 'gu', 'bn', 'pa', 'ur']:  # Indic languages
            return 'hi'
        else:
            return 'unknown'
    except:
        return "unknown"

def is_hindi_text(text: str) -> bool:
    """
    Check if text contains Hindi characters
    """
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(hindi_pattern.search(text))

def is_english_text(text: str) -> bool:
    """
    Check if text contains English characters
    """
    english_pattern = re.compile(r'[a-zA-Z]')
    return bool(english_pattern.search(text))

def get_language_specific_config(language: str, config: Dict) -> Dict:
    """
    Get language-specific configuration from the main config
    """
    if language in config.get('languages', {}):
        return config['languages'][language]
    return config

def create_language_tokenizer(model_name: str, language: str):
    """
    Create appropriate tokenizer for the given language
    """
    from transformers import AutoTokenizer
    
    if language == 'hi':
        # Use Indic-BERT for Hindi
        return AutoTokenizer.from_pretrained('ai4bharat/indic-bert', use_fast=True)
    else:
        # Use the specified model for English
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def validate_multilingual_data(data_path: str) -> Dict[str, int]:
    """
    Validate multilingual dataset and return language distribution
    """
    import pandas as pd
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        # Handle other formats
        return {"error": "Unsupported format"}
    
    language_counts = {}
    for text in df['text']:
        lang = detect_language(str(text))
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    return language_counts

def create_language_specific_datasets(data_path: str, output_dir: str):
    """
    Split multilingual dataset into language-specific files
    """
    import pandas as pd
    import os
    
    df = pd.read_csv(data_path)
    
    # Add language column if not present
    if 'language' not in df.columns:
        df['language'] = df['text'].apply(detect_language)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split by language
    for language in df['language'].unique():
        if language != 'unknown':
            lang_df = df[df['language'] == language]
            output_path = os.path.join(output_dir, f"{language}_data.csv")
            lang_df.to_csv(output_path, index=False)
            print(f"Created {output_path} with {len(lang_df)} examples")
    
    return output_dir
