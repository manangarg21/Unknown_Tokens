import argparse
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import pandas as pd

from src.utils.io import load_yaml, ensure_dir
from src.data.preprocess import TokenizeCollator
from src.utils.metrics import compute_classification_metrics
from src.models.gru import GRUHead
from src.models.rcnn import RCNNHead
from src.models.lora import attach_lora
from src.models.gating import GatingNetwork
from src.utils.embeddings import SentenceEmbeddingEncoder
from src.utils.langid import weak_langid_batch


class BackboneWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        return self.head(seq, attention_mask)


def build_base_model(config: Dict[str, Any], model_type: str) -> Tuple[BackboneWithHead, AutoTokenizer]:
    if model_type == "hinglish" and config["model"].get("use_alt_backbone", False):
        model_name = config["model"]["alt_backbone"]
    else:
        model_name = config["model"]["backbone"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    hidden_size = backbone.config.hidden_size

    if config["model"]["lora"]["enabled"]:
        backbone = attach_lora(
            backbone,
            r=config["model"]["lora"]["r"],
            alpha=config["model"]["lora"]["alpha"],
            dropout=config["model"]["lora"]["dropout"],
        )

    if model_type == "hinglish":
        head = GRUHead(
            hidden_size=hidden_size,
            num_labels=config["data"]["num_labels"],
            hidden=config["model"]["gru"]["hidden"],
            layers=config["model"]["gru"]["layers"],
            bidirectional=config["model"]["gru"].get("bidirectional", True),
            dropout=config["model"]["gru"]["dropout"],
        )
    elif model_type == "english":
        head = RCNNHead(
            hidden_size=hidden_size,
            num_labels=config["data"]["num_labels"],
            conv_channels=config["model"]["rcnn"]["conv_channels"],
            kernel_sizes=tuple(config["model"]["rcnn"]["kernel_sizes"]),
            rnn_hidden=config["model"]["rcnn"]["rnn_hidden"],
            rnn_layers=config["model"]["rcnn"]["rnn_layers"],
            dropout=config["model"]["rcnn"]["dropout"],
        )
    else:
        raise ValueError("model_type must be 'hinglish' or 'english'")

    return BackboneWithHead(backbone, head), tokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Mixture-of-Experts with gating and language auxiliary task")
    ap.add_argument("--hinglish_config", type=str, required=True)
    ap.add_argument("--english_config", type=str, required=True)
    ap.add_argument("--hinglish_weights", type=str, default=None)
    ap.add_argument("--english_weights", type=str, default=None)
    # Train files: allow either single combined or two separate files
    ap.add_argument("--train_file", type=str, default=None)
    ap.add_argument("--train_file_eng", type=str, default=None)
    ap.add_argument("--train_file_hin", type=str, default=None)
    ap.add_argument("--val_file_eng", type=str, required=True)
    ap.add_argument("--val_file_hin", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_lang", type=float, default=0.2)
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/LaBSE")
    ap.add_argument("--freeze_backbones", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Enable MPS fallback on macOS and select best device (cuda > mps > cpu)
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ensure_dir(args.output_dir)

    cfg_hin = load_yaml(args.hinglish_config)
    cfg_eng = load_yaml(args.english_config)

    model_hin, tok_hin = build_base_model(cfg_hin, "hinglish")
    model_eng, tok_eng = build_base_model(cfg_eng, "english")

    # Load pre-trained weights for base models if provided
    if args.hinglish_weights and os.path.isfile(args.hinglish_weights):
        model_hin.load_state_dict(torch.load(args.hinglish_weights, map_location="cpu"))
    else:
        print(f"[MoE] Hinglish weights not found or not provided: {args.hinglish_weights}. Using pretrained backbone only.")
    if args.english_weights and os.path.isfile(args.english_weights):
        model_eng.load_state_dict(torch.load(args.english_weights, map_location="cpu"))
    else:
        print(f"[MoE] English weights not found or not provided: {args.english_weights}. Using pretrained backbone only.")

    if args.freeze_backbones:
        for p in model_hin.parameters():
            p.requires_grad = False
        for p in model_eng.parameters():
            p.requires_grad = False

    model_hin.to(device)
    model_eng.to(device)

    # Gating: sentence embeddings
    embedder = SentenceEmbeddingEncoder(model_name=args.embed_model, device=str(device))
    embed_dim = embedder.encode_texts(["hello"]).shape[-1]
    gate = GatingNetwork(input_dim=int(embed_dim), hidden_dim=256, num_experts=2, num_langs=3).to(device)

    # Data
    if args.train_file is not None:
        df_train = pd.read_csv(args.train_file).rename(columns=lambda x: x.strip())
        df_train = df_train.rename(columns={df_train.columns[0]: "text", df_train.columns[1]: "label"})
    else:
        assert args.train_file_eng is not None and args.train_file_hin is not None, "Provide either --train_file or both --train_file_eng and --train_file_hin"
        df_eng = pd.read_csv(args.train_file_eng).rename(columns=lambda x: x.strip())
        df_hin = pd.read_csv(args.train_file_hin).rename(columns=lambda x: x.strip())
        df_eng = df_eng.rename(columns={df_eng.columns[0]: "text", df_eng.columns[1]: "label"})
        df_hin = df_hin.rename(columns={df_hin.columns[0]: "text", df_hin.columns[1]: "label"})
        df_train = pd.concat([df_eng, df_hin], ignore_index=True)
    df_eng_val = pd.read_csv(args.val_file_eng).rename(columns=lambda x: x.strip())
    df_hin_val = pd.read_csv(args.val_file_hin).rename(columns=lambda x: x.strip())
    df_eng_val = df_eng_val.rename(columns={df_eng_val.columns[0]: "text", df_eng_val.columns[1]: "label"})
    df_hin_val = df_hin_val.rename(columns={df_hin_val.columns[0]: "text", df_hin_val.columns[1]: "label"})

    collate_hin = TokenizeCollator(tokenizer=tok_hin, max_length=cfg_hin["model"]["max_length"])
    collate_eng = TokenizeCollator(tokenizer=tok_eng, max_length=cfg_eng["model"]["max_length"])

    class MixedDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels) -> None:
            self.texts = list(texts)
            self.labels = list(map(int, labels))
            self.lang_ids = weak_langid_batch(self.texts)

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return {"text": self.texts[idx], "label": self.labels[idx], "lang_id": self.lang_ids[idx]}

    train_ds = MixedDataset(df_train["text"].astype(str), df_train["label"])

    def collate_moe(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        lang_ids = torch.tensor([b["lang_id"] for b in batch], dtype=torch.long)
        toks_hin = collate_hin([{ "text": t, "label": 0 } for t in texts])
        toks_eng = collate_eng([{ "text": t, "label": 0 } for t in texts])
        with torch.no_grad():
            embeds = embedder.encode_texts(texts)
        return {
            "eng": toks_eng,
            "hin": toks_hin,
            "labels": labels,
            "lang_ids": lang_ids,
            "embeds": embeds,
        }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_moe)

    ce_lang = nn.CrossEntropyLoss()
    nll = nn.NLLLoss()
    opt = torch.optim.AdamW([
        {"params": gate.parameters(), "lr": args.lr},
        {"params": model_hin.parameters(), "lr": args.lr},
        {"params": model_eng.parameters(), "lr": args.lr},
    ], lr=args.lr)

    model_hin.train(not args.freeze_backbones)
    model_eng.train(not args.freeze_backbones)
    gate.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            labels = batch["labels"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            embeds = batch["embeds"].to(device)
            eng = {k: v.to(device) for k, v in batch["eng"].items() if isinstance(v, torch.Tensor)}
            hin = {k: v.to(device) for k, v in batch["hin"].items() if isinstance(v, torch.Tensor)}

            logits_eng = model_eng(eng["input_ids"], eng["attention_mask"])  # [B, C]
            logits_hin = model_hin(hin["input_ids"], hin["attention_mask"])  # [B, C]

            gate_out = gate(embeds)  # weights: [B, 2], lang_logits: [B, 3]
            w = gate_out["expert_weights"]
            lang_logits = gate_out["lang_logits"]

            # Mixture logits: convex combination
            probs_eng = torch.softmax(logits_eng, dim=-1)
            probs_hin = torch.softmax(logits_hin, dim=-1)
            mix_probs = w[:, 0:1] * probs_eng + w[:, 1:2] * probs_hin
            log_mix_probs = torch.log(mix_probs.clamp_min(1e-8))

            loss_sarcasm = nll(log_mix_probs, labels)
            loss_lang = ce_lang(lang_logits, lang_ids)
            loss = loss_sarcasm + args.lambda_lang * loss_lang

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().cpu())
            pbar.set_postfix({"loss": epoch_loss / (pbar.n + 1)})

        torch.save({
            "gate": gate.state_dict(),
            "model_eng": model_eng.state_dict(),
            "model_hin": model_hin.state_dict(),
        }, os.path.join(args.output_dir, f"moe_epoch_{epoch+1}.pt"))

    # Evaluation helpers
    @torch.no_grad()
    def evaluate(df: pd.DataFrame, name: str) -> None:
        texts = df["text"].astype(str).tolist()
        labels = torch.tensor(df["label"].tolist(), dtype=torch.long).to(device)
        lang_ids = torch.tensor(weak_langid_batch(texts), dtype=torch.long).to(device)

        ds = [{"text": t, "label": 0} for t in texts]
        loader_eng = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_eng(b))
        loader_hin = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_hin(b))

        all_preds = []
        i = 0
        for be, bh in zip(loader_eng, loader_hin):
            be = {k: v.to(device) for k, v in be.items() if isinstance(v, torch.Tensor)}
            bh = {k: v.to(device) for k, v in bh.items() if isinstance(v, torch.Tensor)}
            batch_texts = texts[i:i + be["input_ids"].size(0)]
            i += be["input_ids"].size(0)
            embeds = embedder.encode_texts(batch_texts).to(device)

            logits_eng = model_eng(be["input_ids"], be["attention_mask"])  # [B, C]
            logits_hin = model_hin(bh["input_ids"], bh["attention_mask"])  # [B, C]
            w = gate(embeds)["expert_weights"]
            probs_eng = torch.softmax(logits_eng, dim=-1)
            probs_hin = torch.softmax(logits_hin, dim=-1)
            mix_probs = w[:, 0:1] * probs_eng + w[:, 1:2] * probs_hin
            preds = mix_probs.argmax(dim=-1)
            all_preds.append(preds.cpu())

        y_pred = torch.cat(all_preds, dim=0).tolist()
        y_true = df["label"].tolist()
        metrics = compute_classification_metrics(y_true, y_pred, average="macro" if name == "English" else "weighted")
        print(f"\n--- {name} set metrics ---")
        print(metrics)

    model_hin.eval()
    model_eng.eval()
    gate.eval()

    evaluate(df_eng_val, "English")
    evaluate(df_hin_val, "Hinglish")


if __name__ == "__main__":
    main()


