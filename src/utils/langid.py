from typing import List
import re


LANG_TO_ID = {"en": 0, "hinglish": 1, "hi": 2}


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_HINGLISH_MARKERS = {
    "hai",
    "bhai",
    "nahi",
    "kya",
    "tum",
    "hum",
    "mera",
    "tera",
    "hain",
    "tha",
    "thi",
    "mein",
    "se",
    "ko",
    "kyu",
    "bhi",
}


def _detect_with_cld3(text: str) -> str:
    try:
        import pycld3

        pred = pycld3.get_language(text)
        if pred is None:
            return "en"
        if pred.language == "hi":
            return "hi"
        if pred.language == "en":
            return "en"
        return "en"
    except Exception:
        return "en"


def weak_langid(text: str) -> int:
    t = (text or "").lower()
    if _DEVANAGARI_RE.search(t):
        return LANG_TO_ID["hi"]

    # Hinglish heuristic via common romanized Hindi tokens
    tokens = set(re.findall(r"[a-zA-Z]+", t))
    if tokens and len(tokens & _HINGLISH_MARKERS) > 0:
        return LANG_TO_ID["hinglish"]

    # Fallback to CLD3 if available
    lang = _detect_with_cld3(t)
    return LANG_TO_ID.get(lang, LANG_TO_ID["en"])


def weak_langid_batch(texts: List[str]) -> List[int]:
    return [weak_langid(t) for t in texts]


