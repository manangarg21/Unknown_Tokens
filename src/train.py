import os, argparse
from typing import Dict
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset as HFDataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
import torch
from src.data.datasets import UnifiedSarcasmDataset, DataCollatorWithContext
from src.models.sarcasm_model import SarcasmClassifier, info_nce_loss
from src.utils import set_seed, ensure_special_tokens
from src.commonsense.conceptnet import synthesize_hint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_commonsense", type=int, default=0)
    p.add_argument("--use_contrastive", type=int, default=0)
    p.add_argument("--lambda_contrastive", type=float, default=0.2)
    p.add_argument("--output_dir", type=str, default="outputs/last_run")
    return p.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        probs = torch.tensor(logits).softmax(-1).numpy()[:,1]
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = float("nan")
    return {"macro_f1": macro_f1, "f1_sarcastic": f1, "precision": p, "recall": r, "auroc": auroc}

class SarcasmTrainer(Trainer):
    def __init__(self, lambda_contrastive: float = 0.2, use_contrastive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.use_contrastive = use_contrastive
        self.lambda_contrastive = lambda_contrastive

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        pair_idx = inputs.pop("pair_idx", None)
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss_fct = nn.CrossEntropyLoss()
        ce = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        loss = ce
        if self.use_contrastive and pair_idx is not None:
            proj = outputs["proj"]
            c_loss = info_nce_loss(proj, pair_idx, temperature=0.07)
            loss = ce + self.lambda_contrastive * c_loss
        return (loss, outputs) if return_outputs else loss

def main():
    args = parse_args()
    set_seed(args.seed)

    train_ds = UnifiedSarcasmDataset(args.train_path)
    val_ds = UnifiedSarcasmDataset(args.val_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ensure_special_tokens(tokenizer)

    collator = DataCollatorWithContext(
        tokenizer=tokenizer,
        max_length=args.max_length,
        commonsense_fn=(synthesize_hint if args.use_commonsense else None),
        prepend_commonsense=bool(args.use_commonsense)
    )

    hf_train = HFDataset.from_list([train_ds[i] for i in range(len(train_ds))])
    hf_val = HFDataset.from_list([val_ds[i] for i in range(len(val_ds))])

    model = SarcasmClassifier(args.model_name, num_labels=2)
    model.encoder.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(8, args.batch_size),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
    )

    trainer = SarcasmTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        use_contrastive=bool(args.use_contrastive),
        lambda_contrastive=args.lambda_contrastive
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
