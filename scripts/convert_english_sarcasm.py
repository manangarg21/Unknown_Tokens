#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from typing import List, Tuple
from tqdm.auto import tqdm


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
    return rows


def to_text_label(rows: List[dict], text_key: str, label_key: str) -> List[Tuple[str, int]]:
    converted: List[Tuple[str, int]] = []
    for r in rows:
        if text_key not in r or label_key not in r:
            continue
        text = str(r[text_key]).replace("\n", " ").strip()
        try:
            label = int(r[label_key])
        except Exception:
            # Map common truthy/falsey
            v = r[label_key]
            if isinstance(v, bool):
                label = int(v)
            else:
                try:
                    label = 1 if str(v).lower() in {"1", "true", "sarcastic"} else 0
                except Exception:
                    label = 0
        if text:
            converted.append((text, label))
    return converted


def write_csv(path: str, items: List[Tuple[str, int]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for text, label in items:
            w.writerow([text, int(label)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input1", default="data/english/Sarcasm_Headlines_Dataset.json")
    ap.add_argument("--input2", default="data/english/Sarcasm_Headlines_Dataset_v2.json")
    ap.add_argument("--out_train", default="data/english/train.csv")
    ap.add_argument("--out_val", default="data/english/val.csv")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    rows: List[dict] = []
    if os.path.isfile(args.input1):
        rows.extend(read_jsonl(args.input1))
    if os.path.isfile(args.input2):
        rows.extend(read_jsonl(args.input2))
    if not rows:
        raise SystemExit("No input records found. Ensure input paths are correct.")

    # Progress across conversion
    items = []
    for r in tqdm(rows, desc="Convert", leave=False):
        converted = to_text_label([r], text_key="headline", label_key="is_sarcastic")
        if converted:
            items.append(converted[0])

    # Shuffle and split
    random.shuffle(items)
    n = len(items)
    n_val = max(1, int(n * args.val_ratio))
    val = items[:n_val]
    train = items[n_val:]

    write_csv(args.out_train, train)
    write_csv(args.out_val, val)

    print(f"Wrote {len(train)} train rows -> {args.out_train}")
    print(f"Wrote {len(val)} val rows   -> {args.out_val}")


if __name__ == "__main__":
    main()


