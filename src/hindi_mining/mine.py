from typing import Dict
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def mine_similar(cfg: Dict) -> None:
    # load translated source
    src = pd.read_csv(cfg["embedding"]["translated_src"])  # expects columns: text,text_hi,label,id (id optional)
    index = faiss.read_index(cfg["embedding"]["faiss_index"]) 
    corpus_csv = os.path.splitext(cfg["embedding"]["faiss_index"])[0] + ".csv"
    corpus_df = pd.read_csv(corpus_csv)
    model = SentenceTransformer(cfg["embedding"]["model"])

    mined_rows = []
    top_k = int(cfg["mining"].get("top_k", 3))
    thr = float(cfg["mining"].get("score_threshold", 0.7))
    for _, row in src.iterrows():
        q = row.get("text_hi", row.get("text", ""))
        if not isinstance(q, str) or not q:
            continue
        q_emb = model.encode([q], normalize_embeddings=True)
        D, I = index.search(q_emb.astype(np.float32), top_k)
        for score, idx in zip(D[0], I[0]):
            if score >= thr:
                mined_rows.append({
                    "src_text_en": row.get("text"),
                    "src_text_hi": q,
                    "candidate": corpus_df.iloc[int(idx)]["text"],
                    "score": float(score),
                    "label": row.get("label", None),
                })
    out_df = pd.DataFrame(mined_rows)
    out_df.to_csv(cfg["corpora"]["mined_out"], index=False)
