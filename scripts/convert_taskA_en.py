#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/english/task_A_En_test.csv")
    ap.add_argument("--out_train", default="data/english/taskA_train.csv")
    ap.add_argument("--out_val", default="data/english/taskA_val.csv")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # Normalize headers
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" not in df.columns:
        raise SystemExit("Expected a 'text' column")
    label_col = None
    for c in ["label", "sarcastic", "sarcasm"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise SystemExit("Expected a label column among ['label','sarcastic','sarcasm']")

    out = pd.DataFrame({
        "text": df["text"].astype(str),
        "label": df[label_col].apply(lambda x: 1 if str(x).strip().lower() in {"1","true","yes"} else 0).astype(int)
    })

    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(out)
    n_val = max(1, int(n * args.val_ratio))
    val = out.iloc[:n_val]
    train = out.iloc[n_val:]

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    train.to_csv(args.out_train, index=False)
    val.to_csv(args.out_val, index=False)
    print(f"Wrote {len(train)} train -> {args.out_train}")
    print(f"Wrote {len(val)} val   -> {args.out_val}")


if __name__ == "__main__":
    main()


