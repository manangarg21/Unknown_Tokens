import argparse
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from torch.utils.data import TensorDataset, DataLoader
import joblib

from src.utils.io import load_yaml
from src.utils.metrics import compute_classification_metrics
from src.data.preprocess import load_csv, TokenizeCollator
from src.utils.conceptnet import get_concepts

from src.models.gru import GRUHead
from src.models.rcnn import RCNNHead
from src.models.lora import attach_lora
class Model(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, input_ids, attention_mask, **_: Any):
        # Accept and ignore any extra kwargs (e.g., labels) to avoid passing them to the backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        return self.head(seq, attention_mask)
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a stacking ensemble meta-learner from four separate data files.")
    # --- Model Paths ---
    parser.add_argument("--hinglish_config", type=str, required=True)
    parser.add_argument("--english_config", type=str, required=True)
    parser.add_argument("--hinglish_weights", type=str, required=True)
    parser.add_argument("--english_weights", type=str, required=True)
    # --- Four Separate Data Files ---
    parser.add_argument("--train_file_eng", type=str, required=True, help="Path to the PURE English training CSV.")
    parser.add_argument("--train_file_hin", type=str, required=True, help="Path to the PURE Hinglish training CSV.")
    parser.add_argument("--val_file_eng", type=str, required=True, help="Path to the PURE English validation CSV.")
    parser.add_argument("--val_file_hin", type=str, required=True, help="Path to the PURE Hinglish validation CSV.")
    # --- Other Params ---
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


from typing import Tuple

def load_base_model(config: Dict[str, Any], weights_path: str, model_type: str) -> Tuple[Model, AutoTokenizer]:
    """Loads a pre-trained base model and its tokenizer."""
    print(f"Loading {model_type} model...")
    if model_type == "hinglish" and config["model"].get("use_alt_backbone", False):
        model_name = config["model"]["alt_backbone"]
    else:
        model_name = config["model"]["backbone"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    hidden_size = backbone.config.hidden_size

    if config["model"]["lora"]["enabled"]:
        backbone = attach_lora(backbone, r=config["model"]["lora"]["r"], alpha=config["model"]["lora"]["alpha"], dropout=config["model"]["lora"]["dropout"])

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

    model = Model(backbone, head)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"{model_type.capitalize()} model loaded successfully.")
    return model, tokenizer


@torch.no_grad()
def get_predictions(model: Model, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Generates prediction probabilities from a base model."""
    model.to(device)
    model.eval()
    
    all_probs = []
    for batch in tqdm(loader, desc=f"Getting predictions from {model.__class__.__name__}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        
    return np.vstack(all_probs)


def main():
    args = parse_args()
    # Enable MPS fallback on macOS and select best device (cuda > mps > cpu)
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    seed = 42

    # 1. Load configs and models
    cfg_hin = load_yaml(args.hinglish_config)
    cfg_eng = load_yaml(args.english_config)
    model_hin, tokenizer_hin = load_base_model(cfg_hin, args.hinglish_weights, "hinglish")
    model_eng, tokenizer_eng = load_base_model(cfg_eng, args.english_weights, "english")

    # 2. Load and COMBINE the training datasets
    print("Loading and combining datasets...")
    df_eng_train = pd.read_csv(args.train_file_eng).rename(columns=lambda x: x.strip())
    df_hin_train = pd.read_csv(args.train_file_hin).rename(columns=lambda x: x.strip())
    train_df = pd.concat([df_eng_train, df_hin_train], ignore_index=True)

    # Load validation datasets but KEEP THEM SEPARATE
    df_eng_val = pd.read_csv(args.val_file_eng).rename(columns=lambda x: x.strip())
    df_hin_val = pd.read_csv(args.val_file_hin).rename(columns=lambda x: x.strip())

    # Standardize column names
    train_df = train_df.rename(columns={train_df.columns[0]: 'text', train_df.columns[1]: 'label'})
    df_eng_val = df_eng_val.rename(columns={df_eng_val.columns[0]: 'text', df_eng_val.columns[1]: 'label'})
    df_hin_val = df_hin_val.rename(columns={df_hin_val.columns[0]: 'text', df_hin_val.columns[1]: 'label'})
    
    print(f"Total training samples: {len(train_df)}")
    print(f"English validation samples: {len(df_eng_val)}")
    print(f"Hinglish validation samples: {len(df_hin_val)}")

    print("\n--- Generating features for Meta-Learner ---")
    ds_eng_train = [{"text": t, "label": l} for t, l in zip(train_df["text"], train_df["label"])]
    # Apply ConceptNet enrichment if needed for the Hinglish model's data
    train_text_hin = train_df["text"].astype(str)
    if cfg_hin["model"]["conceptnet"]["enabled"]:
        print("Applying ConceptNet enrichment for Hinglish training data (in parallel)...")
        cache_dir = cfg_hin["model"]["conceptnet"].get("cache_dir")
        max_c = int(cfg_hin["model"]["conceptnet"].get("max_concepts", 5))
        
        # Define the enrichment function (ensure `get_concepts` is imported)
        def enrich_text(t: str) -> str:
            tokens = (t or "").split()
            extra = [c for tok in tokens[:5] for c in get_concepts(tok, cache_dir=cache_dir, max_concepts=max_c)]
            return t + " \n concepts: " + " ".join(extra) if extra else t
        
        # Use joblib to run the function on all available CPU cores
        # n_jobs=-1 means use all cores.
        train_text_hin = Parallel(n_jobs=-1)(
            delayed(enrich_text)(t) for t in tqdm(train_text_hin, desc="Enrich train")
        )
    
    # **THIS IS THE FIX**: Define ds_hin_train from the processed text
    ds_hin_train = [{"text": t, "label": l} for t, l in zip(train_text_hin, train_df["label"])]
    collate_hin = TokenizeCollator(tokenizer=tokenizer_hin, max_length=cfg_hin["model"]["max_length"])
    collate_eng = TokenizeCollator(tokenizer=tokenizer_eng, max_length=cfg_eng["model"]["max_length"])
    
    loader_hin_train = DataLoader(ds_hin_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hin)
    loader_eng_train = DataLoader(ds_eng_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eng)

    preds_hin_train = get_predictions(model_hin, loader_hin_train, device)
    preds_eng_train = get_predictions(model_eng, loader_eng_train, device)
    
    X_train = np.concatenate([preds_hin_train, preds_eng_train], axis=1)
    y_train = train_df["label"].values

    print("\n--- Training PyTorch Meta-Learner on GPU ---")
    print("\n--- Training Meta-Learner ---")
    meta_learner = LogisticRegression(solver='liblinear', random_state=seed)
    meta_learner.fit(X_train, y_train)
    joblib.dump(meta_learner, os.path.join(args.output_dir, "stacking_meta_learner.joblib"))
    print("Meta-learner trained and saved.")


    # 4. Evaluate the Ensemble SEPARATELY
    def evaluate_subset(df_subset, name, method="macro"):
        """Helper function updated for the scikit-learn meta-learner."""
        print(f"\n--- Performance on {name} Validation Set ---")
        
        # Create datasets with and without enrichment
        ds_eng = [{"text": t, "label": l} for t, l in zip(df_subset["text"], df_subset["label"])]
        text_hin = df_subset["text"].astype(str)
        if cfg_hin["model"]["conceptnet"]["enabled"] and enrich_text:
            text_hin = [enrich_text(t) for t in tqdm(text_hin, desc=f"Enrich {name} val")]
        ds_hin = [{"text": t, "label": l} for t, l in zip(text_hin, df_subset["label"])]

        # Create dataloaders
        loader_hin = DataLoader(ds_hin, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hin)
        loader_eng = DataLoader(ds_eng, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eng)

        # Get predictions from base models
        preds_hin = get_predictions(model_hin, loader_hin, device)
        preds_eng = get_predictions(model_eng, loader_eng, device)
        
        # Create feature set for inference
        X_subset = np.concatenate([preds_hin, preds_eng], axis=1)
        y_true = df_subset["label"].values
        
        # Get final predictions using the scikit-learn meta-learner
        final_preds = meta_learner.predict(X_subset)
            
        metrics = compute_classification_metrics(y_true.tolist(), final_preds.tolist(), average=method)
        print(metrics)

    # Evaluate on the English validation set
    evaluate_subset(df_eng_val, "English", method="macro")

    # Evaluate on the Hinglish validation set
    evaluate_subset(df_hin_val, "Hinglish", method="weighted")

if __name__ == "__main__":
    main()