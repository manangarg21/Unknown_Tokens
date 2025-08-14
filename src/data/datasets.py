from typing import Optional, Dict, List, Any
import os, json
import pandas as pd
from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizerBase
from ..utils import join_with_context, detect_language
import re

class MultilingualSarcasmDataset(torch.utils.data.Dataset):
    """
    Multilingual dataset supporting English and Hindi with language detection.
    Expects CSV/JSONL with columns: text, label, optional context, optional pair_id, optional language.
    """
    def __init__(self, path: str, target_language: Optional[str] = None):
        self.items = self._load(path)
        self.target_language = target_language
        
        # Language detection if not provided
        if not self.items[0].get("language"):
            for item in self.items:
                item["language"] = detect_language(item["text"])
        
        # Filter by target language if specified
        if target_language:
            self.items = [item for item in self.items if item["language"] == target_language]
        
        # Setup pair groups for contrastive learning
        self._setup_pairs()
        
        print(f"Loaded {len(self.items)} examples for language: {target_language or 'all'}")

    def _setup_pairs(self):
        """Setup pair indices for contrastive learning"""
        pair_groups = {}
        for i, ex in enumerate(self.items):
            pid = ex.get("pair_id", None)
            if pid is not None and str(pid).strip().upper() != "NA" and str(pid).strip() != "":
                pair_groups.setdefault(str(pid), []).append(i)
        
        pair_idx = [-1] * len(self.items)
        for g in pair_groups.values():
            if len(g) == 2:
                a, b = g
                pair_idx[a] = b
                pair_idx[b] = a
        self.pair_idx = pair_idx

    def _load(self, path: str):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        if path.endswith(".json") or path.endswith(".jsonl"):
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    items.append(json.loads(line))
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            items = df.to_dict(orient="records")
        else:
            raise ValueError("Unsupported file type. Use .csv or .jsonl")
        
        # Normalize and validate data
        norm = []
        for r in items:
            norm.append({
                "text": str(r.get("text", "")),
                "label": int(r.get("label", 0)),
                "context": None if pd.isna(r.get("context", None)) else r.get("context", None),
                "pair_id": r.get("pair_id", None),
                "language": r.get("language", None),
            })
        return norm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        return {
            "text": ex["text"],
            "label": ex["label"],
            "context": ex.get("context", None),
            "pair_idx": self.pair_idx[idx],
            "language": ex["language"],
        }

class UnifiedSarcasmDataset(torch.utils.data.Dataset):
    """
    Backward compatibility wrapper for the original dataset class.
    """
    def __init__(self, path: str):
        self.multilingual_ds = MultilingualSarcasmDataset(path)
        self.items = self.multilingual_ds.items
        self.pair_idx = self.multilingual_ds.pair_idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.multilingual_ds[idx]

@dataclass
class MultilingualDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 192
    commonsense_fn: Optional[Any] = None
    prepend_commonsense: bool = False
    language: Optional[str] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        texts, seconds, labels, pair_idx, languages = [], [], [], [], []
        
        for ex in features:
            text = ex["text"]
            
            # Apply commonsense hints if enabled
            if self.prepend_commonsense and self.commonsense_fn is not None:
                try:
                    hint = self.commonsense_fn(text, ex.get("language", "en"))
                except Exception:
                    hint = ""
                if hint:
                    text = f"[KNOW] {hint} [TEXT] {text}"
            
            # Join with context
            first, second = join_with_context(text, ex.get("context"))
            texts.append(first)
            seconds.append(second)
            labels.append(ex["label"])
            pair_idx.append(ex.get("pair_idx", -1))
            languages.append(ex.get("language", "en"))

        # Tokenize
        enc = self.tokenizer(
            texts, text_pair=seconds, truncation=True, max_length=self.max_length,
            padding=True, return_tensors="pt"
        )
        
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        enc["pair_idx"] = torch.tensor(pair_idx, dtype=torch.long)
        enc["languages"] = languages
        
        return enc

@dataclass
class DataCollatorWithContext:
    """
    Backward compatibility wrapper for the original data collator.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 192
    commonsense_fn: Optional[Any] = None
    prepend_commonsense: bool = False

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        collator = MultilingualDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            commonsense_fn=self.commonsense_fn,
            prepend_commonsense=self.prepend_commonsense
        )
        return collator(features)
