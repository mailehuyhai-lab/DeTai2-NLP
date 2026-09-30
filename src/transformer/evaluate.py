"""Evaluate transformer models and compare with baseline.

Usage: python src/transformer/evaluate.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.transformer.config import (
    BASE_DIR,
    BATCH_SIZE,
    DATA_DIR,
    MAX_LENGTH,
    MODELS_DIR,
    RESULTS_DIR,
    SENTIMENT_MAP,
    TOPIC_MAP,
)
from src.transformer.dataset import FeedbackDataset, load_split
from src.transformer.metrics import (
    compute_metrics,
    confusion_matrix_df,
    per_class_report,
    predictions_df,
)

BASELINE_DIR = BASE_DIR / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "id": [b["id"] for b in batch],
        "text": [b["text"] for b in batch],
    }


def evaluate_model(task: str, label_map: dict) -> dict:
    """Load best checkpoint, evaluate on test."""
    device = torch.device("cpu")
    model_dir = MODELS_DIR / task
    num_labels = len(label_map)
    label_names = [label_map[i] for i in range(num_labels)]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    test_df = load_split(DATA_DIR / "test.csv")
    test_ds = FeedbackDataset(test_df, tokenizer, "text_clean", task, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, collate_fn=collate_fn)

    all_preds, all_labels, all_ids, all_texts = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_ids.extend(batch["id"])
            all_texts.extend(batch["text"])

    metrics = compute_metrics(all_labels, all_preds, label_names)
    report = per_class_report(all_labels, all_preds, label_names)
    cm = confusion_matrix_df(all_labels, all_preds, label_names)
    preds_df = predictions_df(all_ids, all_texts, all_labels, all_preds, label_map)

    return {
        "task": task,
        "metrics": metrics,
        "report": report,
        "cm": cm,
        "preds_df": preds_df,
    }


def compare_with_baseline(transformer_results: list[dict]):
    """Compare PhoBERT with Linear SVM baseline."""
    svm_files = {
        "sentiment": BASELINE_DIR / "sentiment_linear_svm_test_predictions.csv",
        "topic": BASELINE_DIR / "topic_linear_svm_test_predictions.csv",
    }
    label_maps = {"sentiment": SENTIMENT_MAP, "topic": TOPIC_MAP}

    comparison = []
    for tr in transformer_results:
        task = tr["task"]
        svm_df = pd.read_csv(svm_files[task])
        label_map = label_maps[task]
        label_names = [label_map[i] for i in range(len(label_map))]

        svm_metrics = compute_metrics(
            svm_df["true_label"].tolist(),
            svm_df["predicted_label"].tolist(),
            label_names,
        )
        # Per-class F1 for minority classes
        from sklearn.metrics import f1_score

        svm_f1_per = f1_score(
            svm_df["true_label"], svm_df["predicted_label"],
            average=None, zero_division=0,
        )
        pho_f1_per = f1_score(
            tr["preds_df"]["true_label"], tr["preds_df"]["predicted_label"],
            average=None, zero_division=0,
        )

        row = {
            "task": task,
            "metric": "accuracy",
            "SVM": svm_metrics["accuracy"],
            "PhoBERT": tr["metrics"]["accuracy"],
        }
        comparison.append(row)
        for m in ("macro_f1", "weighted_f1"):
            comparison.append({
                "task": task,
                "metric": m,
                "SVM": svm_metrics[m],
                "PhoBERT": tr["metrics"][m],
            })
        for i, name in enumerate(label_names):
            comparison.append({
                "task": task,
                "metric": f"f1_{name}",
                "SVM": round(float(svm_f1_per[i]), 4),
                "PhoBERT": round(float(pho_f1_per[i]), 4),
            })

    comp_df = pd.DataFrame(comparison)
    comp_df["diff"] = (comp_df["PhoBERT"] - comp_df["SVM"]).round(4)
    comp_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return comp_df


def main():
    results = []
    for task, lmap in [("sentiment", SENTIMENT_MAP), ("topic", TOPIC_MAP)]:
        res = evaluate_model(task, lmap)
        results.append(res)

    comp = compare_with_baseline(results)
    print("\n=== Comparison ===")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
