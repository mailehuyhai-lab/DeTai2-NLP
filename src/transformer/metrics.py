"""Metrics computation for transformer evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, label_names: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "macro_recall": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "macro_f1": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "weighted_f1": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
    }


def per_class_report(y_true, y_pred, label_names: list[str]) -> str:
    return classification_report(
        y_true, y_pred, target_names=label_names, digits=4, zero_division=0
    )


def confusion_matrix_df(y_true, y_pred, label_names: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=label_names, columns=label_names)


def predictions_df(
    ids, texts, y_true, y_pred, label_map: dict
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ids,
            "text": texts,
            "true_label": y_true,
            "predicted_label": y_pred,
            "correct": [1 if t == p else 0 for t, p in zip(y_true, y_pred)],
            "true_label_name": [label_map[t] for t in y_true],
            "predicted_label_name": [label_map[p] for p in y_pred],
        }
    )
