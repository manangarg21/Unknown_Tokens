# Unknown_Tokens: Context-Aware Sarcasm and Irony Detection

End-to-end implementation per the outline:
- English track: RoBERTa + RCNN head (+ optional LoRA adapters)
- Hinglish track: mBERT/IndicBERT + GRU head (+ ConceptNet enrichment, + optional LoRA)
- Ensemble of English + Hinglish predictions
- Hindi augmentation/mining: translate EN sarcasm → HI, embed, FAISS similarity search over monolingual corpora to mine sarcastic candidates

All parts are configurable.

## 1) Prerequisites
- Python 3.10–3.12
- GPU with CUDA recommended (CPU works but is slower)
- Internet access for first-time model downloads (Hugging Face)

## 2) Setup
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Data preparation
Place CSVs with two columns `text,label`.
- English: `data/english/train.csv`, `data/english/val.csv`
- Hinglish: `data/hinglish/train.csv`, `data/hinglish/val.csv`

Example CSV (UTF-8):
```csv
text,label
"Great, another power cut in summer",1
"I just love waiting in traffic",1
"The app works as expected.",0
```
- Labels: 0 = non-sarcastic, 1 = sarcastic (binary)
- Adjust paths/columns in the YAML configs under `configs/` if your layout differs.

Directory layout example:
```
Project/
  data/
    english/
      train.csv
      val.csv
    hinglish/
      train.csv
      val.csv
  corpora/          # optional, for Hindi mining
    hindi/
      *.txt         # plain text files, one or more lines per file
```

### 3a) Recommended datasets

Use the following sources to acquire datasets, then convert them to the simple `text,label` CSV format and place them under `data/english/` or `data/hinglish/` as shown above.

- **English**
  - Tweets with Sarcasm and Irony (news headlines): `kaggle datasets download -d rmisra/news-headlines-dataset-for-sarcasm-detection`
  - iSarcasm (intended sarcasm) dataset overview: `paperswithcode.com/dataset/isarcasm` (download via authors' link)
  - SemEval-2018 Task 3: Irony detection in English tweets: `aclanthology.org/S18-1005/` (download via task page/CodaLab)

- **Hinglish**
  - HackArena Multilingual Sarcasm Detection: `kaggle datasets download -d nikhilmaram/hackarena-multilingual-sarcasm-detection`
  - Sarcasm Detection Code-Mixed Dataset: `github.com/nikhilmaram/Sarcasm-Detection-Code-Mixed-Dataset`
  - Sarcasm Detection in Hindi-English Code-Mixed Data: `github.com/nikhilmaram/Sarcasm-Detection-in-Hindi-English-Code-Mixed-Data`

- **Hindi corpora (optional, for mining/augmentation)**
  - IndicCorpv2 (Hindi subset): `ai4bharat.iitm.ac.in/indiccorp`
  - IITB English–Hindi Parallel Text (BPCC): `cfilt.iitb.ac.in/iitb_parallel/`
  - OSCAR dataset (Hindi subset): `oscar-corpus.com`

Notes:
- Ensure each CSV has exactly two columns named `text,label` with labels as 0/1.
- For Kaggle datasets, you can use the Kaggle CLI (after configuring your API token) and unzip locally, then normalize column names before copying to `data/...`.
- Some sources distribute multiple files/splits; you may merge/split as needed to produce `train.csv` and `val.csv`.

### 3b) One-click download (optional)

Use the helper script to fetch and organize public datasets mentioned above. You must have Kaggle CLI configured (`~/.kaggle/kaggle.json`).

```bash
bash scripts/download_datasets.sh
```

The script downloads into `data/english/`, `data/hinglish/`, and prints instructions for manually hosted datasets (SemEval Task 3, iSarcasm authors' link, Hindi corpora).

## 4) English track (RoBERTa + RCNN)
Config: `configs/english_roberta_rcnn.yaml`
- Key toggles:
  - `model.backbone`: e.g., `roberta-base`
  - `model.lora.enabled`: true/false
  - `train.batch_size`, `train.epochs`, etc.

Train:
```powershell
python -m src.train_english --config configs/english_roberta_rcnn.yaml
```
Outputs:
- Best checkpoint → `outputs/english_roberta_rcnn/best.pt`
- Metrics printed each epoch (precision/recall/F1, macro)

## 5) Hinglish track (mBERT/IndicBERT + GRU + ConceptNet)
Config: `configs/hinglish_mbert_gru.yaml`
- Key toggles:
  - `model.backbone`: `bert-base-multilingual-cased`
  - `model.alt_backbone`: `ai4bharat/indic-bert`
  - `model.use_alt_backbone`: switch between the two
  - `model.conceptnet.enabled`: enable semantic enrichment via ConceptNet
  - `model.conceptnet.cache_dir`: on-disk cache for ConceptNet queries
  - `model.lora.enabled`: true/false

Train:
```powershell
python -m src.train_hinglish --config configs/hinglish_mbert_gru.yaml
```
Outputs:
- Best checkpoint → `outputs/hinglish_mbert_gru/best.pt`
- Metrics printed each epoch

ConceptNet notes:
- Requires internet to fetch related concepts the first time
- Results are cached under `cache/conceptnet/` by default
- You can disable by setting `model.conceptnet.enabled: false`

## Hindi dataset generation (guide)
For the end-to-end Hindi dataset creation workflow (translation + mining + curation), see the guide in:
- `src/hindi_dataset_generation/README.md`
