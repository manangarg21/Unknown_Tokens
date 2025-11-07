# Unknown_Tokens: Context-Aware Sarcasm Detection

Context-aware sarcasm/irony detection for English and Hinglish with configurable backbones, lightweight heads, optional LoRA adapters, ConceptNet enrichment, and ensembling/MoE.

- English: RoBERTa or DeBERTa + RCNN/GRU heads (+ optional LoRA, + optional auxiliary heads)
- Hinglish: mBERT/IndicBERT + GRU head (+ ConceptNet enrichment, + optional LoRA)
- Ensembling: weighted average of English + Hinglish predictions
- Mixture-of-Experts (MoE): learned gating over English/Hinglish experts

All parts are driven by YAML configs under `configs/`.

## 1) Prerequisites
- Python 3.10–3.13
- PyTorch with CUDA if available (CPU and Apple MPS are supported; MPS fallback is auto-enabled)
- Internet access for first-time model downloads (Hugging Face)

## 2) Setup
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- If you manage PyTorch separately (e.g., CUDA wheels), install torch first, then run `pip install -r requirements.txt` without the torch line.
- On macOS, MPS is auto-enabled; unsupported ops fall back to CPU.

## 3) Data preparation
The training/eval scripts expect CSVs with two columns `text,label` (UTF-8). Labels: 0 = non-sarcastic, 1 = sarcastic.

Place files as follows (adjust paths in YAML if needed):
- English: `data/english/train.csv`, `data/english/val.csv`
- Hinglish: `data/hinglish/train.csv`, `data/hinglish/val.csv`

Example CSV:
```csv
text,label
"Great, another power cut in summer",1
"I just love waiting in traffic",1
"The app works as expected.",0
```

Optional corpora for Hindi mining/augmentation:
```
Project/
  data/
    english/{train.csv,val.csv}
    hinglish/{train.csv,val.csv}
  corpora/
    hindi/*.txt   # raw text files for mining
```

### 3a) Dataset helpers and converters
Use any of these to assemble `text,label` CSVs:

- Kaggle news headlines (English sarcasm):
  ```bash
  kaggle datasets download -d rmisra/news-headlines-dataset-for-sarcasm-detection -p data/english
  python scripts/convert_english_sarcasm.py --out_train data/english/train.csv --out_val data/english/val.csv
  ```

- SemEval-2018 Task 3 (English tweets):
  ```bash
  python scripts/convert_semeval2018_taskA.py \
    --train_txt data/english/datasets/train/SemEval2018-T3-train-taskA.txt \
    --gold_test_txt data/english/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt \
    --input_test_txt data/english/datasets/test_TaskA/SemEval2018-T3_input_test_taskA.txt \
    --out_dir data/english/semeval
  ```

## 4) Training

### 4.1 English (RoBERTa + RCNN)
Config: `configs/english_roberta_rcnn.yaml`
```bash
python -m src.train_english --config configs/english_roberta_rcnn.yaml
```
Outputs:
- Best checkpoint: `outputs/english_roberta_rcnn/best.pt`

Alternative English config with auxiliary heads (SemEval): `configs/english_roberta_rcnn_semeval.yaml`
- Adds `aux_heads.sentiment` and `aux_heads.flip` on pooled RCNN features.
- Toggle LoRA with `model.lora.enabled`.

DeBERTa + GRU variant:
```bash
python -m src.train_english --config configs/english_deberta_gru.yaml
```

### 4.2 Hinglish (mBERT/IndicBERT + GRU + ConceptNet)
Config: `configs/hinglish_mbert_gru.yaml`
```bash
python -m src.train_hinglish --config configs/hinglish_mbert_gru.yaml
```
Notes:
- ConceptNet enrichment can be disabled via `model.conceptnet.enabled: false`.
- Related concepts are cached under `cache/conceptnet/`.
- Switch to IndicBERT with `model.use_alt_backbone: true`.

## 5) Evaluation / Inference

### 5.1 Single model (English or Hinglish)
Use the generic tester for any config that defines either RCNN or GRU head:
```bash
python -m src.test_english \
  --config configs/english_roberta_rcnn.yaml \
  --csv data/english/val.csv \
  --ckpt outputs/english_roberta_rcnn/best.pt \
  --out_preds outputs/preds/test_preds.csv \
  --out_metrics outputs/preds/test_metrics.json
```
Column names are auto-detected; override with `--text_col/--label_col` if needed.

Hinglish tester (writes predictions and optional metrics if labels present):
```bash
python -m src.test_hinglish \
  --config configs/hinglish_mbert_gru.yaml \
  --test_file data/hinglish/val.csv \
  --output_file outputs/preds/hinglish_val_preds.csv
```

## 6) Configuration quick reference
Common keys across YAML files in `configs/`:
- `model.backbone` / `model.alt_backbone` / `model.use_alt_backbone`
- `model.max_length`
- `model.lora.enabled|r|alpha|dropout`
- Head-specific:
  - RCNN: `model.rcnn.conv_channels|kernel_sizes|rnn_hidden|rnn_layers|dropout`
  - GRU: `model.gru.hidden|layers|bidirectional|dropout`
- Optional aux heads (English RCNN): `aux_heads.sentiment`, `aux_heads.flip`
- Training: `train.batch_size|epochs|lr|weight_decay|warmup_ratio|grad_accum_steps|fp16`
- Data: `data.train_file|val_file|text_col|label_col|num_labels`

## 7) SLURM helper scripts (optional)
- English training job: see `run_english.sh` (creates a venv on the node, installs deps, runs `src.train_english` with the SemEval config)

## 8) Tips & troubleshooting
- macOS: MPS is used when available; unsupported ops fall back to CPU automatically.
- First runs download models/tokenizers to the Hugging Face cache; ensure internet access.
- If checkpoints fail to load with/without LoRA, the ensemble tester will retry toggling LoRA automatically; otherwise ensure the `model.lora.enabled` flag matches training.
- ConceptNet enrichment can slow data prep on first epoch; it is cached afterward.

## 9) Web app for sentence prediction

A minimal FastAPI app is provided under `webapp/` to predict sarcasm for a single sentence and show results in a browser.

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the web server (default port 8000)
uvicorn webapp.main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

Environment overrides (optional):

```bash
# Use a different config or checkpoint if desired
export SARCASM_CONFIG=configs/english_roberta_rcnn_sarcasm.yaml
export SARCASM_CKPT=outputs/english_roberta_rcnn_sarcasm/best.pt
```

## 10) Model Outputs

All models are present in this google drive: [Drive Link](https://drive.google.com/drive/folders/1CQxew9L6jNaGJyISBQn8ZjFrxPpu2APl?usp=sharing)

