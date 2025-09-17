import argparse
from typing import Dict, Any
import os
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_global_seed
from src.utils.metrics import compute_classification_metrics
from src.utils.conceptnet import get_concepts
from src.data.preprocess import load_csv, TokenizeCollator
from src.models.gru import GRUHead
from src.models.lora import attach_lora


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def main(cfg: Dict[str, Any]) -> None:
    set_global_seed(cfg.get("seed", 42))
    accelerator = Accelerator()
    ensure_dir(cfg["output_dir"]) 

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

    class Model(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, input_ids, attention_mask, **_: Any):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            seq = outputs.last_hidden_state
            return self.head(seq, attention_mask)

    model = Model(backbone, head)

    # Data
    train_df = load_csv(cfg["data"]["train_file"], cfg["data"]["text_col"], cfg["data"]["label_col"]).rename(columns={cfg["data"]["text_col"]:"text", cfg["data"]["label_col"]:"label"})
    val_df = load_csv(cfg["data"]["val_file"], cfg["data"]["text_col"], cfg["data"]["label_col"]).rename(columns={cfg["data"]["text_col"]:"text", cfg["data"]["label_col"]:"label"})

    # ConceptNet enrichment: naive token-level feature injection via appending top concepts to text
    if cfg["model"]["conceptnet"]["enabled"]:
        cache_dir = cfg["model"]["conceptnet"].get("cache_dir")
        max_c = int(cfg["model"]["conceptnet"].get("max_concepts", 5))
        def enrich_text(t: str) -> str:
            tokens = (t or "").split()
            extra = []
            for tok in tokens[:5]:
                extra.extend(get_concepts(tok, cache_dir=cache_dir, max_concepts=max_c))
            if extra:
                return t + " \n concepts: " + " ".join(extra)
            return t
        train_df["text"] = train_df["text"].astype(str).map(enrich_text)
        val_df["text"] = val_df["text"].astype(str).map(enrich_text)

    collate = TokenizeCollator(tokenizer=tokenizer, max_length=cfg["model"]["max_length"]) 

    class DS(torch.utils.data.Dataset):
        def __init__(self, df): self.df = df
        def __len__(self): return len(self.df)
        def __getitem__(self, idx): return {"text": self.df.iloc[idx]["text"], "label": int(self.df.iloc[idx]["label"]) }

    train_loader = DataLoader(DS(train_df), batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(DS(val_df), batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)

    # Optim
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    num_steps = len(train_loader) * cfg["train"]["epochs"] // max(1, cfg["train"].get("grad_accum_steps", 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * cfg["train"]["warmup_ratio"]), num_steps)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    best_f1 = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        train_iter = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Train E{epoch+1}")
        for step, batch in enumerate(train_iter, 1):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
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
                torch.save(unwrapped.state_dict(), os.path.join(cfg["output_dir"], "best.pt"))


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
