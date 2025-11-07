from typing import Dict, List
import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from src.utils.io import ensure_dir


def _read_corpus_lines(mono_dir: str) -> List[str]:
    lines: List[str] = []
    for fp in glob.glob(os.path.join(mono_dir, "**", "*.txt"), recursive=True):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        lines.append(ln)
        except Exception:
            continue
    return lines


def build_index(cfg: Dict) -> None:
    ensure_dir(os.path.dirname(cfg["embedding"]["faiss_index"]))
    mono_dir = cfg["corpora"]["mono_dir"]
    lines = _read_corpus_lines(mono_dir)
    model = SentenceTransformer(cfg["embedding"]["model"])
    embs = model.encode(lines, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = int(cfg["embedding"].get("dim", embs.shape[1]))
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    faiss.write_index(index, cfg["embedding"]["faiss_index"]) 
    # save corpus
    pd.DataFrame({"text": lines}).to_csv(os.path.splitext(cfg["embedding"]["faiss_index"])[0] + ".csv", index=False)
