import argparse
from src.utils.io import load_yaml, ensure_dir
from src.hindi_mining.translate import translate_dataset
from src.hindi_mining.embed_index import build_index
from src.hindi_mining.mine import mine_similar


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src_csv", required=False, help="English sarcasm CSV with columns: text,label,id")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    ensure_dir(cfg["output_dir"]) 
    # 1) translation (optional if src_csv is provided)
    if args.src_csv:
        translate_dataset(args.src_csv, cfg)
    # 2) embeddings + index over monolingual corpus
    build_index(cfg)
    # 3) mining
    mine_similar(cfg)


if __name__ == "__main__":
    main()
