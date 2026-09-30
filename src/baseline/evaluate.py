"""
evaluate.py — Evaluation functions for baseline models

Metrics: Accuracy, Precision/Recall/F1 (macro + weighted),
         Classification report, Confusion matrix.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, label_names=None) -> dict:
    """
    Tính toàn bộ metrics cho 1 task/model.

    Args:
        y_true: nhãn thực tế
        y_pred: nhãn dự đoán
        label_names: tên nhãn (cho classification report)

    Returns:
        dict chứa tất cả metrics
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def get_classification_report(y_true, y_pred, label_names=None, labels=None) -> str:
    """
    Tạo classification report text.

    Args:
        y_true: nhãn thực tế
        y_pred: nhãn dự đoán
        label_names: tên nhãn hiển thị
        labels: danh sách label values (đảm bảo thứ tự đúng)

    Returns:
        Classification report string
    """
    return classification_report(
        y_true, y_pred,
        target_names=label_names,
        labels=labels,
        zero_division=0,
    )


def get_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """
    Tạo confusion matrix.

    Args:
        y_true: nhãn thực tế
        y_pred: nhãn dự đoán
        labels: danh sách label values (đảm bảo thứ tự đúng)

    Returns:
        Confusion matrix array
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def confusion_matrix_to_df(cm, label_names) -> pd.DataFrame:
    """
    Chuyển confusion matrix array thành DataFrame có label rõ ràng.

    Args:
        cm: confusion matrix numpy array
        label_names: tên nhãn

    Returns:
        DataFrame với index=True labels, columns=Predicted labels
    """
    index_names = [f"True_{name}" for name in label_names]
    col_names = [f"Pred_{name}" for name in label_names]
    return pd.DataFrame(cm, index=index_names, columns=col_names)


def create_predictions_df(ids, texts, y_true, y_pred, label_map=None) -> pd.DataFrame:
    """
    Tạo DataFrame predictions cho error analysis sau này.

    Args:
        ids: danh sách id
        texts: danh sách text gốc
        y_true: nhãn thực tế
        y_pred: nhãn dự đoán
        label_map: dict mapping int → tên nhãn (optional)

    Returns:
        DataFrame với cột: id, text, true_label, predicted_label, correct
    """
    df = pd.DataFrame({
        "id": ids,
        "text": texts,
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": [int(t == p) for t, p in zip(y_true, y_pred)],
    })

    if label_map:
        df["true_label_name"] = df["true_label"].map(label_map)
        df["predicted_label_name"] = df["predicted_label"].map(label_map)

    return df
