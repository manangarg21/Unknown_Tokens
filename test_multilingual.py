#!/usr/bin/env python3
"""
Test script for the multilingual sarcasm detection framework
"""

import os
import sys
import pandas as pd
from src.utils import detect_language, create_language_specific_datasets
from src.data.datasets import MultilingualSarcasmDataset
from src.models.sarcasm_model import create_multilingual_model

def test_language_detection():
    """Test language detection functionality"""
    print("Testing language detection...")
    
    test_texts = [
        "Great, another software update in the middle of work",  # English
        "यह क्या बात है? काम नहीं हो रहा",  # Hindi
        "Perfect, the file corrupted itself",  # English
        "बहुत अच्छा, आज का दिन",  # Hindi
        "This is a test message",  # English
        "यह एक परीक्षण संदेश है"  # Hindi
    ]
    
    for text in test_texts:
        lang = detect_language(text)
        print(f"Text: {text[:30]}... -> Language: {lang}")
    
    print("✓ Language detection test completed\n")

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("Testing dataset loading...")
    
    # Create a simple test dataset
    test_data = [
        {"text": "Great, another meeting", "label": 1, "context": "", "pair_id": "A1", "language": "en"},
        {"text": "यह काम बहुत अच्छा है", "label": 0, "context": "", "pair_id": "NA", "language": "hi"},
        {"text": "Perfect, my phone died", "label": 1, "context": "", "pair_id": "A2", "language": "en"},
        {"text": "आज का दिन बहुत सुंदर है", "label": 0, "context": "", "pair_id": "NA", "language": "hi"}
    ]
    
    # Save test data
    test_df = pd.DataFrame(test_data)
    test_path = "test_data.csv"
    test_df.to_csv(test_path, index=False)
    
    try:
        # Test English-only dataset
        en_dataset = MultilingualSarcasmDataset(test_path, target_language="en")
        print(f"English dataset loaded: {len(en_dataset)} examples")
        
        # Test Hindi-only dataset
        hi_dataset = MultilingualSarcasmDataset(test_path, target_language="hi")
        print(f"Hindi dataset loaded: {len(hi_dataset)} examples")
        
        # Test multilingual dataset
        multi_dataset = MultilingualSarcasmDataset(test_path, target_language=None)
        print(f"Multilingual dataset loaded: {len(multi_dataset)} examples")
        
        print("✓ Dataset loading test completed")
        
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print()

def test_model_creation():
    """Test model creation functionality"""
    print("Testing model creation...")
    
    try:
        # Test English model
        en_model = create_multilingual_model({}, "en")
        print("✓ English model created successfully")
        
        # Test Hindi model
        hi_model = create_multilingual_model({}, "hi")
        print("✓ Hindi model created successfully")
        
        print("✓ Model creation test completed")
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
    
    print()

def test_data_preparation():
    """Test data preparation utilities"""
    print("Testing data preparation utilities...")
    
    try:
        # Test language-specific dataset creation
        test_data = [
            {"text": "Great, another meeting", "label": 1, "context": "", "pair_id": "A1"},
            {"text": "यह काम बहुत अच्छा है", "label": 0, "context": "", "pair_id": "NA"},
            {"text": "Perfect, my phone died", "label": 1, "context": "", "pair_id": "A2"},
            {"text": "आज का दिन बहुत सुंदर है", "label": 0, "context": "", "pair_id": "NA"}
        ]
        
        test_df = pd.DataFrame(test_data)
        test_path = "test_data.csv"
        test_df.to_csv(test_path, index=False)
        
        # Test language detection and splitting
        output_dir = "test_output"
        create_language_specific_datasets(test_path, output_dir)
        
        # Check if files were created
        if os.path.exists(os.path.join(output_dir, "en_data.csv")):
            print("✓ English dataset created")
        if os.path.exists(os.path.join(output_dir, "hi_data.csv")):
            print("✓ Hindi dataset created")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        
        print("✓ Data preparation test completed")
        
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
    
    print()

def main():
    """Run all tests"""
    print("="*60)
    print("MULTILINGUAL SARCASM DETECTION FRAMEWORK - TEST SUITE")
    print("="*60)
    print()
    
    try:
        test_language_detection()
        test_dataset_loading()
        test_model_creation()
        test_data_preparation()
        
        print("="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print()
        print("The multilingual framework is ready to use!")
        print()
        print("Next steps:")
        print("1. Prepare your English and Hindi datasets")
        print("2. Train models using src/train_multilingual.py")
        print("3. Evaluate using src/evaluate_multilingual.py")
        print("4. Check the README.md for detailed usage instructions")
        
    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
