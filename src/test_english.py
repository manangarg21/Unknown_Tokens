import argparse
import json
import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from src.utils.io import load_yaml
from src.utils.metrics import compute_classification_metrics
from src.data.preprocess import load_csv, TokenizeCollator
from src.models.rcnn import RCNNHead
from src.models.gru import GRUHead
from src.models.lora import attach_lora


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True, help="CSV with columns text,label")
    ap.add_argument("--ckpt", type=str, required=False, help="Path to model state_dict .pt")
    ap.add_argument("--out_preds", type=str, default="outputs/preds/test_preds.csv")
    ap.add_argument("--out_metrics", type=str, default="outputs/preds/test_metrics.json")
    ap.add_argument("--text_col", type=str, required=False, help="Override text column name in CSV")
    ap.add_argument("--label_col", type=str, required=False, help="Override label column name in CSV")
    return ap.parse_args()


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_name = cfg["model"].get("alt_backbone") if cfg["model"].get("use_alt_backbone", False) else cfg["model"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)

    if cfg["model"]["lora"]["enabled"]:
        backbone = attach_lora(
            backbone,
            r=cfg["model"]["lora"]["r"],
            alpha=cfg["model"]["lora"]["alpha"],
            dropout=cfg["model"]["lora"]["dropout"],
        )

    hidden_size = backbone.config.hidden_size
    head: nn.Module
    if "rcnn" in cfg["model"]:
        head = RCNNHead(
            hidden_size=hidden_size,
            num_labels=cfg["data"]["num_labels"],
            conv_channels=cfg["model"]["rcnn"]["conv_channels"],
            kernel_sizes=tuple(cfg["model"]["rcnn"]["kernel_sizes"]),
            rnn_hidden=cfg["model"]["rcnn"]["rnn_hidden"],
            rnn_layers=cfg["model"]["rcnn"]["rnn_layers"],
            dropout=cfg["model"]["rcnn"]["dropout"],
        )
    elif "gru" in cfg["model"]:
        head = GRUHead(
            hidden_size=hidden_size,
            num_labels=cfg["data"]["num_labels"],
            hidden=cfg["model"]["gru"]["hidden"],
            layers=cfg["model"]["gru"]["layers"],
            bidirectional=cfg["model"]["gru"].get("bidirectional", True),
            dropout=cfg["model"]["gru"]["dropout"],
        )
    else:
        raise ValueError("Config must contain either model.rcnn or model.gru")

    # Match the attribute names used during training ("backbone" and "head") so
    # the saved state_dict keys align and load without renaming.
    class M(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, input_ids, attention_mask, **kwargs):
            seq = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.head(seq, attention_mask)

    return M(backbone, head), tokenizer


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # Enable MPS fallback on macOS and select best device (cuda > mps > cpu)
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model, tokenizer = build_model(cfg)
    model.to(device)
    model.eval()

    # Load checkpoint
    ckpt = args.ckpt or os.path.join(cfg["output_dir"], "best.pt")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)

    # Data
    # Load with flexible column handling
    import pandas as pd
    raw = pd.read_csv(args.csv)
    cols = {c.lower(): c for c in raw.columns}
    # Determine text/label columns: CLI override > config > common aliases
    text_candidates = [
        (args.text_col or ""),
        cfg["data"].get("text_col", ""),
        "text",
        "tweet",
        "headline",
    ]
    label_candidates = [
        (args.label_col or ""),
        cfg["data"].get("label_col", ""),
        "label",
        "sarcastic",
        "sarcasm",
    ]
    text_col = next((cols[x.lower()] for x in text_candidates if x and x.lower() in cols), None)
    label_col = next((cols[x.lower()] for x in label_candidates if x and x.lower() in cols), None)
    if not text_col:
        raise SystemExit("Could not find a text column (tried overrides and aliases: text/tweet/headline)")
    if not label_col:
        raise SystemExit("Could not find a label column (try --label_col or use one of: label/sarcastic/sarcasm)")
    df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()
    
    collate = TokenizeCollator(tokenizer=tokenizer, max_length=cfg["model"]["max_length"]) 

    class DS(Dataset):
        def __init__(self, df): self.df = df
        def __len__(self): return len(self.df)
        def __getitem__(self, idx): return {"text": self.df.iloc[idx]["text"], "label": int(self.df.iloc[idx]["label"])}

    loader = DataLoader(DS(df), batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)

    # Eval
    os.makedirs(os.path.dirname(args.out_preds), exist_ok=True)
    import numpy as np
    ids, p0, p1, preds, labels = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Test")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            y = batch["labels"].cpu().numpy()
            for j in range(prob.shape[0]):
                ids.append(i * loader.batch_size + j)
                p0.append(float(prob[j,0]))
                p1.append(float(prob[j,1]))
                preds.append(int(np.argmax(prob[j])))
                labels.append(int(y[j]))

    # Save predictions
    import pandas as pd
    out_df = pd.DataFrame({"id": ids, "p0": p0, "p1": p1, "pred": preds, "label": labels})
    out_df.to_csv(args.out_preds, index=False)

    # Metrics
    metrics = compute_classification_metrics(labels, preds)
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print({"ckpt": ckpt, **metrics, "out_preds": args.out_preds})


if __name__ == "__main__":
    main()


