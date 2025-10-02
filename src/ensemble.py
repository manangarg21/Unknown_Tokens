import argparse
from typing import List
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=False, help="List of CSV files with columns: id,p0,p1")
    ap.add_argument("--weights", nargs="+", type=float, required=False, help="Same length as preds")
    ap.add_argument("--out", type=str, default="outputs/ensemble.csv")
    return ap.parse_args()


def weighted_average(prob_mats: List[np.ndarray], weights: List[float]) -> np.ndarray:
    w = np.array(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    out = np.zeros_like(prob_mats[0])
    for pm, wi in zip(prob_mats, w):
        out += wi * pm
    return out


def main():
    args = parse_args()
    if not args.preds or not args.weights:
        print("Provide --preds and --weights for ensembling.")
        return
    prob_mats = []
    ids = None
    for p in args.preds:
        df = pd.read_csv(p)
        if ids is None:
            ids = df["id"].values
        prob_mats.append(df[["p0", "p1"]].values.astype(np.float32))
    probs = weighted_average(prob_mats, args.weights)
    out_df = pd.DataFrame({"id": ids, "p0": probs[:,0], "p1": probs[:,1], "pred": probs.argmax(axis=1)})
    out_df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
