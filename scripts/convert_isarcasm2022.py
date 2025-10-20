#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import pandas as pd
from tqdm.auto import tqdm


def detect_label_column(df: pd.DataFrame) -> str:
    for cand in ["sarcastic", "sarcasm", "label"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not find a label column among ['sarcastic','sarcasm','label']")


def load_isarcasm_csv(path: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    # Drop unnamed index columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "tweet" not in df.columns:
        raise ValueError("Expected a 'tweet' column in the dataset")
    label_col = detect_label_column(df)
    # Clean
    df = df[["tweet", label_col]].dropna()
    # Cast labels to int 0/1
    df[label_col] = df[label_col].apply(lambda x: 1 if str(x).strip().lower() in {"1","true","yes"} else 0)
    return df["tweet"], df[label_col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/english/isarcasm2022.csv")
    ap.add_argument("--out_train", default="data/english/isarcasm_train.csv")
    ap.add_argument("--out_val", default="data/english/isarcasm_val.csv")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    texts, labels = load_isarcasm_csv(args.input)
    df = pd.DataFrame({"text": texts.astype(str), "label": labels.astype(int)})
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * args.val_ratio))
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    train_df.to_csv(args.out_train, index=False)
    val_df.to_csv(args.out_val, index=False)
    print(f"Wrote {len(train_df)} train rows -> {args.out_train}")
    print(f"Wrote {len(val_df)} val rows   -> {args.out_val}")


if __name__ == "__main__":
    main()


