# Multilingual Sarcasm and Irony Detection Framework

A flexible and robust framework for detecting sarcasm and irony in **English** and **Hindi** text, leveraging contextual analysis and transformer-based models.

## 🎯 Project Overview

This project addresses the challenge of detecting non-literal language (sarcasm and irony) in multilingual settings. Unlike traditional sentiment analysis, sarcasm detection requires understanding context, discourse structure, and pragmatic interpretation.

### Key Features

- **Multilingual Support**: Native support for English and Hindi with language-specific models
- **Context-Aware Analysis**: Incorporates conversational context and discourse-level features
- **Modular Architecture**: Easy extension to new languages, domains, and data types
- **Advanced Modeling**: Transformer-based models with contrastive learning and commonsense integration
- **Comprehensive Evaluation**: Language-specific metrics and detailed error analysis

## 🏗️ Architecture

```
src/
├── models/                 # Model architectures
│   └── sarcasm_model.py   # Multilingual classifier with language embeddings
├── data/                  # Dataset handling
│   ├── datasets.py        # Multilingual dataset classes
│   ├── prepare_examples.py # Data preparation utilities
│   └── prepare_hindi_data.py # Hindi-specific data preparation
├── commonsense/           # External knowledge integration
│   └── conceptnet.py      # ConceptNet commonsense hints
├── train_multilingual.py  # Multilingual training script
├── evaluate_multilingual.py # Multilingual evaluation script
└── utils.py               # Language detection and utilities
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd common_sense_sarcasm_project
pip install -r requirements.txt
```

### 2. Data Preparation

#### English Datasets
```bash
# Convert SARC dataset
python src/data/prepare_examples.py --task sarc --inp data/sarc.csv --out data/sarc_unified.csv

# Convert Headlines dataset
python src/data/prepare_examples.py --task headlines --inp data/headlines.json --out data/headlines_unified.csv
```

#### Hindi Datasets
```bash
# Create synthetic Hindi data for testing
python src/data/prepare_hindi_data.py --task synthetic --output data/hindi_synthetic.csv --num_examples 1000

# Convert Hindi Twitter data
python src/data/prepare_hindi_data.py --task twitter --input data/hindi_twitter.csv --output data/hindi_unified.csv
```

### 3. Training

#### English-Only Model
```bash
python src/train_multilingual.py \
    --config configs/default.yaml \
    --train_path data/english_train.csv \
    --val_path data/english_val.csv \
    --language en \
    --output_dir outputs/english_model
```

#### Hindi-Only Model
```bash
python src/train_multilingual.py \
    --config configs/default.yaml \
    --train_path data/hindi_train.csv \
    --val_path data/hindi_val.csv \
    --language hi \
    --output_dir outputs/hindi_model
```

#### Multilingual Model (Both Languages)
```bash
python src/train_multilingual.py \
    --config configs/default.yaml \
    --train_path data/combined_train.csv \
    --val_path data/combined_val.csv \
    --language both \
    --output_dir outputs/multilingual_model
```

### 4. Evaluation

```bash
python src/evaluate_multilingual.py \
    --model_dir outputs/multilingual_model \
    --val_path data/combined_val.csv \
    --language both \
    --output_dir evaluation_results
```

## 📊 Supported Datasets

### English
- **SARC**: Reddit comments with sarcasm labels and context
- **Headlines**: News headlines labeled as sarcastic
- **iSarcasm**: Twitter pairs (sarcastic/non-sarcastic)

### Hindi
- **Custom Hindi Twitter**: Hindi sarcasm dataset
- **Synthetic Data**: Generated Hindi examples for testing
- **Mixed Code**: Hinglish (Hindi-English mixed) text

## 🔧 Configuration

The framework uses YAML configuration files for easy customization:

```yaml
# configs/default.yaml
model_name: roberta-base
batch_size: 16
epochs: 3
lr: 2e-5

languages:
  en:
    model_name: roberta-base
    max_length: 192
    use_commonsense: true
    use_contrastive: true
  hi:
    model_name: ai4bharat/indic-bert
    max_length: 256
    use_commonsense: false
    use_contrastive: true
```

## 🧠 Model Architecture

### MultilingualSarcasmClassifier
- **Language-Specific Encoders**: Uses appropriate models for each language
  - English: RoBERTa, BERT, or custom models
  - Hindi: Indic-BERT or other Indic language models
- **Language Embeddings**: Explicit language identification for better cross-lingual understanding
- **Attention Visualization**: Built-in attention weight extraction for interpretability
- **Contrastive Learning**: InfoNCE loss for better representations

### EnsembleMultilingualClassifier
- **Multiple Models**: Separate encoders for each language
- **Intelligent Routing**: Automatically routes inputs to appropriate language model
- **Fusion Layer**: Combines representations when needed

## 📈 Training Features

- **Contrastive Learning**: Uses pair information for better representations
- **Commonsense Integration**: Optional ConceptNet hints for English
- **Context Handling**: Supports parent comments and conversation threads
- **Language Detection**: Automatic language identification and routing
- **Gradient Accumulation**: Efficient training with large models

## 📊 Evaluation Metrics

- **Overall Performance**: Macro F1, Precision, Recall, AUROC
- **Language-Specific**: Separate metrics for each language
- **Detailed Analysis**: Confusion matrices and error analysis
- **Prediction Confidence**: Sarcasm probability scores
- **Attention Visualization**: Word-level attention weights

## 🌐 Language Support

### English
- Full commonsense integration (ConceptNet)
- Context-aware processing
- Advanced transformer models

### Hindi
- Indic-BERT backbone
- Hindi-specific preprocessing
- Cultural context awareness

### Future Extensions
- **Multilingual**: Support for more Indic languages
- **Code-Mixed**: Hinglish and other mixed language text
- **Domain Adaptation**: Product reviews, social media, etc.

## 🔍 Interpretability

- **Attention Weights**: Visualize which words the model focuses on
- **Language Embeddings**: Understand language-specific representations
- **Context Analysis**: See how context influences predictions
- **Error Analysis**: Detailed breakdown of misclassifications

## 📁 Data Format

The framework expects a unified CSV format:

```csv
text,label,context,pair_id,language
"Great, another meeting",1,"Previous context here",A1,en
"यह काम बहुत अच्छा है",0,"",NA,hi
```

## 🚀 Advanced Usage

### Custom Models
```python
from src.models.sarcasm_model import create_multilingual_model

# Create custom model
model = create_multilingual_model(config, language="hi")
```

### Custom Datasets
```python
from src.data.datasets import MultilingualSarcasmDataset

# Load custom dataset
dataset = MultilingualSarcasmDataset("path/to/data.csv", target_language="en")
```

### Language Detection
```python
from src.utils import detect_language

# Detect language of text
lang = detect_language("यह क्या बात है?")
print(lang)  # Output: "hi"
```

## 📚 Related Papers

- **SARC Dataset**: "A Large Self-Annotated Corpus for Sarcasm" (Khodak et al., 2018)
- **iSarcasm**: "iSarcasm: A Dataset of Intended Sarcasm" (Abu Farha et al., 2021)
- **Indic-BERT**: "IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages" (Kakwani et al., 2020)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ConceptNet for commonsense knowledge
- HuggingFace for transformer models
- Indic-BERT team for Hindi language models
- The sarcasm detection research community

---

**For questions and support, please open an issue or contact the maintainers.**
