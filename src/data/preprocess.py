from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
from transformers import PreTrainedTokenizer


def load_csv(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    return df


@dataclass
class TokenizeCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = [int(item["label"]) for item in batch]
        toks = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        toks["labels"] = __import__("torch").tensor(labels, dtype=__import__("torch").long)
        return toks
