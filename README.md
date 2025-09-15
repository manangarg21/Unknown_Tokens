# Unknown_Tokens: Context-Aware Sarcasm and Irony Detection

End-to-end implementation per the outline:
- English track: RoBERTa + RCNN head (+ optional LoRA adapters)
- Hinglish track: mBERT/IndicBERT + GRU head (+ ConceptNet enrichment, + optional LoRA)
- Ensemble of English + Hinglish predictions
- Optional Hindi augmentation/mining: translate EN sarcasm → HI, embed, FAISS similarity search over monolingual corpora to mine sarcastic candidates

All optional parts are included and configurable.

## 1) Prerequisites
- Python 3.10–3.12
- GPU with CUDA recommended (CPU works but is slower)
- Internet access for first-time model downloads (Hugging Face)

## 2) Setup
From the project root:
```powershell
# Create and activate a virtual environment (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```
Bash alternative:
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

## 6) Optional: Generate prediction CSVs for ensembling
The ensemble expects probability CSVs of the form: `id,p0,p1` (+ optional `pred`). If you want to ensemble dev/validation predictions, export them with a small script after training. Example snippet (edit paths/model names as needed):
```python
import os, torch, pandas as pd
from transformers import AutoTokenizer, AutoModel
from src.models.rcnn import RCNNHead
from src.data.preprocess import TokenizeCollator
from torch.utils.data import DataLoader, Dataset

# Config
model_name = "roberta-base"
ckpt = "outputs/english_roberta_rcnn/best.pt"
val_csv = "data/english/val.csv"
max_len = 128

# Load
tok = AutoTokenizer.from_pretrained(model_name)
backbone = AutoModel.from_pretrained(model_name)
head = RCNNHead(hidden_size=backbone.config.hidden_size, num_labels=2)
class M(torch.nn.Module):
    def __init__(self,b,h):
        super().__init__(); self.b=b; self.h=h
    def forward(self, ids, mask):
        seq = self.b(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.h(seq, mask)
model = M(backbone, head)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

# Data
df = pd.read_csv(val_csv)
texts, labels = df["text"].tolist(), df["label"].tolist()
collate = TokenizeCollator(tok, max_len)
class DS(Dataset):
    def __len__(self): return len(texts)
    def __getitem__(self, i): return {"text": texts[i], "label": int(labels[i])}
loader = DataLoader(DS(), batch_size=32, shuffle=False, collate_fn=collate)

# Predict
import torch, numpy as np
probs_all, ids = [], list(range(len(texts)))
with torch.no_grad():
    for batch in loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
probs_all = np.vstack(probs_all)

# Save
out = pd.DataFrame({"id": ids, "p0": probs_all[:,0], "p1": probs_all[:,1], "pred": probs_all.argmax(1)})
os.makedirs("outputs/preds", exist_ok=True)
out.to_csv("outputs/preds/english_val_probs.csv", index=False)
```
Repeat similarly for the Hinglish model to produce `outputs/preds/hinglish_val_probs.csv`.

## 7) Ensemble
Weighted average:
```powershell
python -m src.ensemble --preds outputs/preds/english_val_probs.csv outputs/preds/hinglish_val_probs.csv --weights 0.5 0.5 --out outputs/preds/ensemble_val.csv
```
- Change the weights to emphasize one track over the other
- The output includes `pred` computed from the combined probabilities

## 8) Optional: Hindi augmentation/mining
Config: `configs/hindi_mining.yaml`
- Inputs:
  - `--src_csv`: English sarcasm CSV with columns `text,label` (and optional `id`)
  - Monolingual Hindi corpora: put `.txt` files under `corpora/hindi/` (nested folders allowed)

Steps:
```powershell
# 1) Translate EN sarcasm to Hindi (writes Sar_H')
python -m src.hindi_mining.run --config configs/hindi_mining.yaml --src_csv data/english/train.csv

# 2) Build FAISS index over monolingual Hindi corpora
# 3) Mine semantically similar candidates
# (both are executed by the same run command once translated file exists)
```
Outputs:
- FAISS index: `outputs/hindi_mining/index.faiss`
- Mined candidates: `outputs/hindi_mining/mined_candidates.csv`

Notes:
- Manually spot-check `mined_candidates.csv` to validate sarcasm quality before training any Hindi model
- If FAISS installation is problematic on Windows, consider using conda (`conda install -c pytorch faiss-cpu`) or run the mining on Linux; you can skip this optional step otherwise

## 9) Configuration tips
- LoRA: toggle `model.lora.enabled` and tune `r`, `alpha`, `dropout`
- Sequence length: `model.max_length`
- Optim schedule: `train.lr`, `train.warmup_ratio`, `train.grad_accum_steps`
- ConceptNet: disable if you prefer pure text baselines or are offline

## 10) Reproducibility
- Global seed set in YAML (`seed`)
- Best checkpoints saved under `outputs/.../best.pt`
- Logs/metrics printed to console; redirect stdout to save

## 11) Troubleshooting
- First run downloads models; ensure internet or pre-cache HF models
- Windows + FAISS: if `pip install faiss-cpu` fails, try conda (`conda install faiss-cpu -c pytorch`) or skip Hindi mining
- CUDA OOM: lower `train.batch_size` or increase `train.grad_accum_steps`, reduce `model.max_length`
- Slow CPU runs: prefer GPU; or enable LoRA to reduce trainable params

## 12) License
Academic/research use.
