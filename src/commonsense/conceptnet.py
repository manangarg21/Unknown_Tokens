# Lightweight ConceptNet retriever with on-disk JSONL cache.
# We fetch a few English relations for content words and synthesize a compact hint string.

import os, re, json
from typing import List, Dict
import requests

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "conceptnet_cache.jsonl")

def _normalize_term(term: str) -> str:
    term = term.lower().strip()
    term = re.sub(r"[^a-z0-9\s_-]", "", term)
    term = re.sub(r"\s+", "_", term)
    return term

def _cache_read_all() -> Dict[str, str]:
    data = {}
    if not os.path.exists(CACHE_FILE):
        return data
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                data[obj["key"]] = obj["value"]
            except Exception:
                continue
    return data

def _cache_append(key: str, value: str):
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

_CACHE = _cache_read_all()

def conceptnet_edges(term: str, max_edges: int = 3) -> List[Dict]:
    term = _normalize_term(term)
    url = f"https://api.conceptnet.io/query?node=/c/en/{term}&other=/c/en&limit={max_edges}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("edges", [])
    except Exception:
        return []
    return []

def synthesize_hint(text: str, max_terms: int = 4) -> str:
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    stop = set("the a an and or but with this that these those for to of from in on at is are was were be have has had you we they he she it not just very really would could should".split())
    cand = [t for t in tokens if t not in stop][:max_terms]
    parts = []
    for t in cand:
        key = f"{t}"
        if key in _CACHE:
            parts.append(_CACHE[key])
            continue
        edges = conceptnet_edges(t, max_edges=3)
        triples = []
        for e in edges:
            rel = e.get("rel", {}).get("label", "")
            end = e.get("end", {}).get("label", "")
            start = e.get("start", {}).get("label", "")
            other = end if start.lower() == t else start
            if rel and other:
                triples.append(f"{t} {rel} {other}")
        hint = "; ".join(triples[:2])
        if hint:
            _CACHE[key] = hint
            _cache_append(key, hint)
            parts.append(hint)
    return "; ".join(parts[:3])
