# Helpers to convert popular datasets into the unified schema.

import json, argparse, pandas as pd

def from_headlines_json(json_path: str, out_csv: str):
    rows = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            rows.append({"text": obj["headline"], "label": int(obj["is_sarcastic"]), "context": "", "pair_id": ""})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def from_sarc_csv(in_csv: str, out_csv: str):
    df = pd.read_csv(in_csv)
    df2 = pd.DataFrame({
        "text": df["comment"],
        "label": df["label"].astype(int),
        "context": df.get("parent", ""),
        "pair_id": ""
    })
    df2.to_csv(out_csv, index=False)

def from_isarcasm_pairs(pairs_csv: str, out_csv: str):
    df = pd.read_csv(pairs_csv)
    rows = []
    for idx, r in df.iterrows():
        gid = r.get("group_id", idx)
        rows.append({"text": r["sarcastic"], "label": 1, "context": "", "pair_id": gid})
        rows.append({"text": r["nonsarcastic"], "label": 0, "context": "", "pair_id": gid})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["headlines","sarc","isarcasm"], required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.task == "headlines":
        from_headlines_json(args.inp, args.out)
    elif args.task == "sarc":
        from_sarc_csv(args.inp, args.out)
    else:
        from_isarcasm_pairs(args.inp, args.out)
