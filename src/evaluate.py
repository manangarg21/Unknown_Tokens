import argparse, numpy as np, torch
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report, roc_auc_score
from src.data.datasets import UnifiedSarcasmDataset, DataCollatorWithContext
from src.models.sarcasm_model import SarcasmClassifier
from src.utils import ensure_special_tokens
from src.commonsense.conceptnet import synthesize_hint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--use_commonsense", type=int, default=0)
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()

    ds = UnifiedSarcasmDataset(args.val_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    ensure_special_tokens(tokenizer)
    model = SarcasmClassifier(args.model_dir, num_labels=2)
    model.eval()

    collator = DataCollatorWithContext(tokenizer, max_length=args.max_length,
                                       commonsense_fn=(synthesize_hint if args.use_commonsense else None),
                                       prepend_commonsense=bool(args.use_commonsense))

    texts, seconds, labels = [], [], []
    for i in range(len(ds)):
        ex = ds[i]
        text = ex["text"]
        if args.use_commonsense:
            try:
                hint = synthesize_hint(text)
                if hint:
                    text = f"[KNOW] {hint} [TEXT] {text}"
            except Exception:
                pass
        if ex.get("context"):
            first, second = "[PARENT] " + ex["context"], "[TEXT] " + text
        else:
            first, second = "[TEXT] " + text, None
        texts.append(first); seconds.append(second); labels.append(ex["label"])

    enc = tokenizer(texts, text_pair=seconds, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
    logits = model(**{k:v for k,v in enc.items() if k in ("input_ids","attention_mask","token_type_ids")})["logits"]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(-1)
    print(classification_report(np.array(labels), preds, digits=4))
    try:
        print("AUROC:", roc_auc_score(np.array(labels), probs[:,1]))
    except Exception:
        pass

if __name__ == "__main__":
    main()
