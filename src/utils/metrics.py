from typing import List, Dict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def compute_classification_metrics(y_true: List[int], y_pred: List[int], y_scores=None, average: str = "macro") -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    metrics = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    if y_scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except Exception:
            pass
    return metrics
