import argparse
from typing import Dict, Any
import os
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim

from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_global_seed
from src.utils.metrics import compute_classification_metrics
from src.data.preprocess import load_csv, TokenizeCollator
from src.models.rcnn import RCNNHead
from src.models.gru import GRUHead
from src.models.lora import attach_lora


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def main(cfg: Dict[str, Any]) -> None:
    set_global_seed(cfg.get("seed", 42))
    # Enable MPS fallback on macOS for unsupported ops
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    mixed_precision = "fp16" if cfg["train"].get("fp16", False) and device_type == "cuda" else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    ensure_dir(cfg["output_dir"]) 

    model_name = cfg["model"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    hidden_size = backbone.config.hidden_size

    if cfg["model"]["lora"]["enabled"]:
        backbone = attach_lora(backbone, r=cfg["model"]["lora"]["r"], alpha=cfg["model"]["lora"]["alpha"], dropout=cfg["model"]["lora"]["dropout"]) 

    # Select head based on config presence (supports either RCNN or GRU)
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

    class Model(nn.Module):
        def __init__(self, backbone, head, aux_heads: Dict[str, nn.Module] | None = None, feature_dim: int | None = None):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.aux_heads = nn.ModuleDict(aux_heads or {})
            self.feature_dim = feature_dim
        def forward(self, input_ids, attention_mask, return_features: bool = False, **_: Any):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            seq = outputs.last_hidden_state
            if hasattr(self.head, "forward_with_features"):
                logits, feats = self.head.forward_with_features(seq, attention_mask)
            else:
                logits = self.head(seq, attention_mask)
                feats = None
            if return_features:
                return logits, feats
            return logits

    # Build auxiliary heads if requested
    aux_defs: Dict[str, nn.Module] = {}
    features_dim: int | None = None
    if hasattr(head, "features_dim"):
        features_dim = int(getattr(head, "features_dim"))
    aux_cfg = cfg.get("aux_heads", {})
    if aux_cfg and features_dim:
        if "sentiment" in aux_cfg:
            aux_defs["sentiment"] = nn.Linear(features_dim, int(aux_cfg["sentiment"].get("num_classes", 3)))
        if "flip" in aux_cfg:
            aux_defs["flip"] = nn.Linear(features_dim, int(aux_cfg["flip"].get("num_classes", 2)))

    model = Model(backbone, head, aux_heads=aux_defs, feature_dim=features_dim)

    # Data (+ optional auxiliary labels)
    train_df = load_csv(cfg["data"]["train_file"], cfg["data"]["text_col"], cfg["data"]["label_col"]).rename(columns={cfg["data"]["text_col"]:"text", cfg["data"]["label_col"]:"label"})
    val_df = load_csv(cfg["data"]["val_file"], cfg["data"]["text_col"], cfg["data"]["label_col"]).rename(columns={cfg["data"]["text_col"]:"text", cfg["data"]["label_col"]:"label"})
    extra_keys: list[str] = []
    sentiment_col = cfg["data"].get("sentiment_col")
    flip_col = cfg["data"].get("flip_col")
    if sentiment_col and sentiment_col in train_df.columns:
        train_df = train_df.rename(columns={sentiment_col: "sentiment"})
        if sentiment_col in val_df.columns:
            val_df = val_df.rename(columns={sentiment_col: "sentiment"})
        extra_keys.append("sentiment")
    if flip_col and flip_col in train_df.columns:
        train_df = train_df.rename(columns={flip_col: "flip"})
        if flip_col in val_df.columns:
            val_df = val_df.rename(columns={flip_col: "flip"})
        extra_keys.append("flip")
    collate = TokenizeCollator(tokenizer=tokenizer, max_length=cfg["model"]["max_length"], extra_label_keys=extra_keys) 

    class DS(torch.utils.data.Dataset):
        def __init__(self, df): self.df = df
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            item = {"text": row["text"], "label": int(row["label"]) }
            if "sentiment" in row:
                try: item["sentiment"] = int(row["sentiment"]) 
                except Exception: pass
            if "flip" in row:
                try: item["flip"] = int(row["flip"]) 
                except Exception: pass
            return item

    train_loader = DataLoader(DS(train_df), batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(DS(val_df), batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)

    # Optim
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    num_steps = len(train_loader) * cfg["train"]["epochs"] // max(1, cfg["train"].get("grad_accum_steps", 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * cfg["train"]["warmup_ratio"]), num_steps)
    criterion = nn.CrossEntropyLoss()
    criterion_aux = nn.CrossEntropyLoss()

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    best_f1 = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        train_iter = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Train E{epoch+1}")
        for step, batch in enumerate(train_iter, 1):
            out = model(batch["input_ids"], batch["attention_mask"], return_features=bool(aux_defs))
            if isinstance(out, tuple):
                logits, feats = out
            else:
                logits, feats = out, None
            loss = criterion(logits, batch["labels"])
            if feats is not None and aux_defs:
                aux_loss = 0.0
                if "sentiment" in aux_defs and "labels_sentiment" in batch:
                    aux_logits = model.aux_heads["sentiment"](feats)
                    w = float(aux_cfg.get("sentiment", {}).get("loss_weight", 0.2))
                    aux_loss = aux_loss + w * criterion_aux(aux_logits, batch["labels_sentiment"])
                if "flip" in aux_defs and "labels_flip" in batch:
                    aux_logits = model.aux_heads["flip"](feats)
                    w = float(aux_cfg.get("flip", {}).get("loss_weight", 0.2))
                    aux_loss = aux_loss + w * criterion_aux(aux_logits, batch["labels_flip"])
                loss = loss + aux_loss
            accelerator.backward(loss)
            if step % cfg["train"].get("grad_accum_steps", 1) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % cfg["logging"]["log_steps"] == 0 and accelerator.is_main_process:
                accelerator.print(f"epoch {epoch+1} step {step}: loss={loss.item():.4f}")
        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            val_iter = tqdm(val_loader, disable=not accelerator.is_local_main_process, desc=f"Val   E{epoch+1}")
            for batch in val_iter:
                logits = model(batch["input_ids"], batch["attention_mask"])
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(accelerator.gather(preds).cpu().tolist())
                all_labels.extend(accelerator.gather(batch["labels"]).cpu().tolist())
        if accelerator.is_main_process:
            metrics = compute_classification_metrics(all_labels, all_preds)
            accelerator.print({"epoch": epoch+1, **metrics})
            f1 = metrics.get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                ensure_dir(cfg["output_dir"]) 
                unwrapped = accelerator.unwrap_model(model)
                state = unwrapped.state_dict()
                filtered = {k: v for k, v in state.items() if k.startswith("backbone.") or k.startswith("head.")}
                torch.save(filtered, os.path.join(cfg["output_dir"], "best.pt"))


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
