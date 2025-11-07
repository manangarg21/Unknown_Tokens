from typing import Dict
import pandas as pd
from transformers import pipeline


def translate_dataset(src_csv: str, cfg: Dict) -> None:
    df = pd.read_csv(src_csv)
    model_name = cfg["translation"]["model"]
    batch_size = int(cfg["translation"].get("batch_size", 16))
    pipe = pipeline("translation", model=model_name)
    out_texts = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        outs = pipe(batch["text"].tolist())
        out_texts.extend([o["translation_text"] for o in outs])
    out = df.copy()
    out["text_hi"] = out_texts
    out.to_csv(cfg["embedding"]["translated_src"], index=False)
