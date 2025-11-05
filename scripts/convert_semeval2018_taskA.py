#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def read_labeled_tsv(path: str) -> pd.DataFrame:
    # SemEval labeled files are tab-separated with header: Tweet index\tLabel\tTweet text
    df = pd.read_csv(path, sep="\t", quoting=3)
    # Normalize headers
    cols = {c.strip().lower(): c for c in df.columns}
    idx_col = next((cols[c] for c in ["tweet index", "tweet_index", "index"] if c in cols), None)
    label_col = next((cols[c] for c in ["label", "gold label", "gold_label"] if c in cols), None)
    text_col = next((cols[c] for c in ["tweet text", "tweet", "text"] if c in cols), None)
    if text_col is None or label_col is None:
        raise SystemExit(f"Could not find expected columns in labeled file: {path} -> have {list(df.columns)}")
    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": df[label_col].astype(int)
    })
    return out


def read_unlabeled_tsv(path: str) -> pd.DataFrame:
    # Test input file: tab-separated with header: tweet index\ttweet text
    df = pd.read_csv(path, sep="\t", quoting=3)
    cols = {c.strip().lower(): c for c in df.columns}
    text_col = next((cols[c] for c in ["tweet text", "tweet", "text"] if c in cols), None)
    if text_col is None:
        raise SystemExit(f"Could not find tweet text column in unlabeled file: {path} -> have {list(df.columns)}")
    out = pd.DataFrame({"text": df[text_col].astype(str)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", type=str, required=True, help="SemEval2018-T3-train-taskA.txt (labeled)")
    ap.add_argument("--gold_test_txt", type=str, required=True, help="SemEval2018-T3_gold_test_taskA_emoji.txt (labeled gold)")
    ap.add_argument("--input_test_txt", type=str, required=True, help="SemEval2018-T3_input_test_taskA.txt (unlabeled input)")
    ap.add_argument("--out_dir", type=str, default="data/english/semeval2018_taskA")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read labeled train and gold test
    train_full = read_labeled_tsv(args.train_txt)
    gold_test = read_labeled_tsv(args.gold_test_txt)

    # Shuffle and split train -> train/val
    train_full = train_full.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(train_full)
    n_val = max(1, int(n * args.val_ratio))
    val = train_full.iloc[:n_val].reset_index(drop=True)
    train = train_full.iloc[n_val:].reset_index(drop=True)

    # Unlabeled input test (for submission-style inference)
    input_test = read_unlabeled_tsv(args.input_test_txt)

    # Write outputs
    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    gold_test_path = os.path.join(args.out_dir, "gold_test.csv")
    input_test_path = os.path.join(args.out_dir, "input_test.csv")
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    gold_test.to_csv(gold_test_path, index=False)
    input_test.to_csv(input_test_path, index=False)

    print(f"Wrote: \n  train -> {train_path} ({len(train)})\n  val   -> {val_path} ({len(val)})\n  gold  -> {gold_test_path} ({len(gold_test)})\n  input -> {input_test_path} ({len(input_test)})")


if __name__ == "__main__":
    main()


