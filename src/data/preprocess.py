from dataclasses import dataclass
from typing import List, Dict, Any
import re
import pandas as pd
from transformers import PreTrainedTokenizer


def load_csv(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    return df


# Emoji removal helpers
# Matches common Unicode emoji blocks and related symbols; also strips ZWJ and variation selectors.
_EMOJI_REGEX = re.compile(
    r"[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    r"\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    r"\U00002700-\U000027BF\U00002600-\U000026FF\U000024C2-\U0001F251]",
    flags=re.UNICODE,
)
_EMOJI_MODIFIERS_REGEX = re.compile(r"[\u200D\uFE0F\U0001F3FB-\U0001F3FF]")
_EMOJI_ALIAS_REGEX = re.compile(r":[a-zA-Z0-9_+\-]+:")


def strip_emojis_from_text(text: str, remove_aliases: bool = True) -> str:
    if not isinstance(text, str):
        return text
    # Remove Unicode emoji and common modifiers
    text = _EMOJI_REGEX.sub("", text)
    text = _EMOJI_MODIFIERS_REGEX.sub("", text)
    # Optionally remove colon-style aliases like :face_with_tears_of_joy:
    if remove_aliases:
        text = _EMOJI_ALIAS_REGEX.sub("", text)
    # Collapse repeated whitespace introduced by removals
    return re.sub(r"\s+", " ", text).strip()


def maybe_strip_emojis_inplace(df: pd.DataFrame, remove_emojis: bool, remove_aliases: bool) -> None:
    if not remove_emojis:
        return
    if "text" not in df.columns:
        return
    df["text"] = [strip_emojis_from_text(t, remove_aliases=remove_aliases) for t in df["text"].astype(str)]


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
