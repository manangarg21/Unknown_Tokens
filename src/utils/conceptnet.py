import os
import json
from typing import List
import requests
from cachetools import LRUCache


_CACHE = LRUCache(maxsize=10000)


def _cn_url(term: str) -> str:
    term = term.replace(" ", "_")
    return f"https://api.conceptnet.io/related/c/en/{term}?filter=/c/en&limit=5"


def get_concepts(token: str, cache_dir: str | None = None, max_concepts: int = 5) -> List[str]:
    key = token.lower()
    if key in _CACHE:
        return _CACHE[key]
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        fp = os.path.join(cache_dir, f"{key}.json")
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _CACHE[key] = data
                    return data
            except Exception:
                pass
    try:
        resp = requests.get(_cn_url(key), timeout=5)
        resp.raise_for_status()
        related = [edge["@id"].split("/")[-1] for edge in resp.json().get("related", [])]
        concepts = related[:max_concepts]
    except Exception:
        concepts = []
    _CACHE[key] = concepts
    if cache_dir is not None:
        try:
            with open(os.path.join(cache_dir, f"{key}.json"), "w", encoding="utf-8") as f:
                json.dump(concepts, f)
        except Exception:
            pass
    return concepts
