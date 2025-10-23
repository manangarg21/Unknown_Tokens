
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel
from src.utils.io import load_yaml
from src.utils.seed import set_global_seed
from src.data.preprocess import load_csv, TokenizeCollator
from src.models.gru import GRUHead
from src.models.lora import attach_lora
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--test_file', type=str, required=True)
    ap.add_argument('--output_file', type=str, required=True)
    return ap.parse_args()

class Model(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, input_ids, attention_mask, **_):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        return self.head(seq, attention_mask)

def main(cfg, test_file, output_file):
    set_global_seed(cfg.get('seed', 42))
    accelerator = Accelerator()
    model_name = cfg["model"]["alt_backbone"] if cfg["model"].get("use_alt_backbone", False) else cfg["model"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    hidden_size = backbone.config.hidden_size
    if cfg["model"]["lora"]["enabled"]:
        backbone = attach_lora(backbone, r=cfg["model"]["lora"]["r"], alpha=cfg["model"]["lora"]["alpha"], dropout=cfg["model"]["lora"]["dropout"])
    head = GRUHead(
        hidden_size=hidden_size,
        num_labels=cfg["data"]["num_labels"],
        hidden=cfg["model"]["gru"]["hidden"],
        layers=cfg["model"]["gru"]["layers"],
        bidirectional=cfg["model"]["gru"].get("bidirectional", True),
        dropout=cfg["model"]["gru"]["dropout"],
    )
    model = Model(backbone, head)
    # Load best checkpoint
    best_ckpt = os.path.join(cfg["output_dir"], "best.pt")
    model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))
    model.eval()
    model = accelerator.prepare(model)
    # Prepare test data
    # Try to load label_col if present
    label_col = cfg["data"].get("label_col", None)
    test_df = load_csv(test_file, cfg["data"]["text_col"], label_col).rename(columns={cfg["data"]["text_col"]: "text"})
    collate = TokenizeCollator(tokenizer=tokenizer, max_length=cfg["model"]["max_length"])
    class TestDS(torch.utils.data.Dataset):
        def __init__(self, df): self.df = df
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            item = {"text": self.df.iloc[idx]["text"]}
            if "label" in self.df.columns:
                item["label"] = int(self.df.iloc[idx]["label"])
            return item
    test_loader = DataLoader(TestDS(test_df), batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            logits = model(batch["input_ids"].to(accelerator.device), batch["attention_mask"].to(accelerator.device))
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            if "labels" in batch:
                all_labels.extend(batch["labels"].cpu().tolist())
    # Save predictions
    test_df["pred_label"] = all_preds
    test_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")
    # If ground truth labels are present, compute metrics
    if "label" in test_df.columns:
        acc = accuracy_score(test_df["label"], test_df["pred_label"])
        f1 = f1_score(test_df["label"], test_df["pred_label"], average="weighted")
        prec = precision_score(test_df["label"], test_df["pred_label"], average="weighted", zero_division=0)
        rec = recall_score(test_df["label"], test_df["pred_label"], average="weighted")
        print(f"Accuracy: {acc:.4f}\nF1: {f1:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}")

if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config)
    main(cfg, args.test_file, args.output_file)
