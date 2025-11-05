import argparse
import json
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import numpy as np

from src.utils.io import load_yaml
from src.utils.metrics import compute_classification_metrics
from src.data.preprocess import TokenizeCollator
from src.test_english import build_model as build_single_model
from copy import deepcopy


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_a", type=str, required=True)
    ap.add_argument("--config_b", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True, help="CSV with columns text,label (or provide overrides)")
    ap.add_argument("--ckpt_a", type=str, default=None)
    ap.add_argument("--ckpt_b", type=str, default=None)
    ap.add_argument("--weight_a", type=float, default=0.2, help="Weight for model A (0-1). Model B gets 1 - weight_a")
    ap.add_argument("--out_preds", type=str, default="outputs/preds/test_preds_ensemble.csv")
    ap.add_argument("--out_metrics", type=str, default="outputs/preds/test_metrics_ensemble.json")
    ap.add_argument("--text_col", type=str, required=False)
    ap.add_argument("--label_col", type=str, required=False)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_a = load_yaml(args.config_a)
    cfg_b = load_yaml(args.config_b)

    # Device
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    def build_and_load(cfg: Dict[str, Any], ckpt: str) -> Tuple[nn.Module, Any]:
        """Build model/tokenizer from cfg; if state_dict mismatches LoRA, retry toggling LoRA."""
        def try_load(cand_cfg: Dict[str, Any]) -> Tuple[nn.Module, Any, bool]:
            model, tok = build_single_model(cand_cfg)
            model.to(device).eval()
            state = torch.load(ckpt, map_location="cpu")
            try:
                model.load_state_dict(state)
                return model, tok, True
            except RuntimeError as e:
                # Heuristic: if missing keys mention base_model.model or lora, likely LoRA mismatch
                msg = str(e).lower()
                if "base_model.model" in msg or "lora" in msg:
                    return model, tok, False
                raise

        # First attempt: as-is
        model, tok, ok = try_load(cfg)
        if ok:
            return model, tok
        # Retry by toggling LoRA enabled flag
        cfg2 = deepcopy(cfg)
        if "lora" in cfg2.get("model", {}):
            cfg2["model"]["lora"]["enabled"] = not bool(cfg2["model"]["lora"]["enabled"])
        model2, tok2, ok2 = try_load(cfg2)
        if ok2:
            return model2, tok2
        # Final fallback: strict=False load (not recommended, but prevents crash)
        model_fallback, tok_fallback = build_single_model(cfg)
        model_fallback.to(device).eval()
        state_fb = torch.load(ckpt, map_location="cpu")
        model_fallback.load_state_dict(state_fb, strict=False)
        return model_fallback, tok_fallback

    # Build two models and load checkpoints robustly
    ckpt_a = args.ckpt_a or os.path.join(cfg_a["output_dir"], "best.pt")
    ckpt_b = args.ckpt_b or os.path.join(cfg_b["output_dir"], "best.pt")
    model_a, tok_a = build_and_load(cfg_a, ckpt_a)
    model_b, tok_b = build_and_load(cfg_b, ckpt_b)

    # Data loading with flexible columns (align with test_english)
    import pandas as pd
    raw = pd.read_csv(args.csv)
    cols = {c.lower(): c for c in raw.columns}
    text_candidates = [
        (args.text_col or ""),
        cfg_a["data"].get("text_col", ""),
        cfg_b["data"].get("text_col", ""),
        "text",
        "tweet",
        "headline",
    ]
    label_candidates = [
        (args.label_col or ""),
        cfg_a["data"].get("label_col", ""),
        cfg_b["data"].get("label_col", ""),
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

    # Two collators (may differ across tokenizers)
    collate_a = TokenizeCollator(tokenizer=tok_a, max_length=cfg_a["model"]["max_length"])
    collate_b = TokenizeCollator(tokenizer=tok_b, max_length=cfg_b["model"]["max_length"])

    class DS(Dataset):
        def __init__(self, df): self.df = df
        def __len__(self): return len(self.df)
        def __getitem__(self, idx): return {"text": self.df.iloc[idx]["text"], "label": int(self.df.iloc[idx]["label"]) }

    loader_a = DataLoader(DS(df), batch_size=cfg_a["train"]["batch_size"], shuffle=False, collate_fn=collate_a)
    loader_b = DataLoader(DS(df), batch_size=cfg_b["train"]["batch_size"], shuffle=False, collate_fn=collate_b)

    # Ensure same number of steps and batch sizes
    if len(loader_a) != len(loader_b):
        raise SystemExit("Batch alignment mismatch between A and B. Use same batch_size in both configs.")

    w_a = float(args.weight_a)
    w_b = 1.0 - w_a

    ids, p0, p1, preds, labels = [], [], [], [], []
    with torch.no_grad():
        for i, (ba, bb) in enumerate(tqdm(zip(loader_a, loader_b), total=len(loader_a), desc="Ensemble Test")):
            ia = ba["input_ids"].to(device); ma = ba["attention_mask"].to(device)
            ib = bb["input_ids"].to(device); mb = bb["attention_mask"].to(device)
            la = ba["labels"].to(device)

            logits_a = model_a(input_ids=ia, attention_mask=ma)
            logits_b = model_b(input_ids=ib, attention_mask=mb)
            prob_a = torch.softmax(logits_a, dim=-1)
            prob_b = torch.softmax(logits_b, dim=-1)
            prob = w_a * prob_a + w_b * prob_b

            y = la.cpu().numpy()
            prob_np = prob.cpu().numpy()
            for j in range(prob_np.shape[0]):
                ids.append(i * loader_a.batch_size + j)
                p0.append(float(prob_np[j, 0]))
                p1.append(float(prob_np[j, 1]))
                preds.append(int(np.argmax(prob_np[j])))
                labels.append(int(y[j]))

    # Save predictions
    import pandas as pd
    out_df = pd.DataFrame({"id": ids, "p0": p0, "p1": p1, "pred": preds, "label": labels})
    os.makedirs(os.path.dirname(args.out_preds), exist_ok=True)
    out_df.to_csv(args.out_preds, index=False)

    # Metrics
    metrics = compute_classification_metrics(labels, preds)
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print({"weight_a": w_a, **metrics, "out_preds": args.out_preds})


if __name__ == "__main__":
    main()


