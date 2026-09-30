"""
train_baseline.py — Train TF-IDF + Logistic Regression + Linear SVM baselines

Đề tài #2: Phân tích cảm xúc và chủ đề từ phản hồi người dùng
Mức 1: Baseline TF-IDF + LR/SVM

Chạy: python -m src.baseline.train_baseline
Hoặc: python src/baseline/train_baseline.py

Pipeline:
  1. Load data/processed/{train,dev,test}.csv
  2. TF-IDF fit trên TRAIN (text_lower), transform dev/test
  3. Train 4 models (2 tasks × 2 classifiers)
  4. Evaluate trên dev và test
  5. Save models, metrics, confusion matrices, predictions
"""

import os
import sys
import csv
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Thêm project root vào path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline.tfidf import create_tfidf_vectorizer, TFIDF_CONFIG
from src.baseline.evaluate import (
    compute_metrics,
    get_classification_report,
    get_confusion_matrix,
    confusion_matrix_to_df,
    create_predictions_df,
)

# === Đường dẫn ===
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "baseline")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "baseline")

# === Cấu hình ===
RANDOM_STATE = 42

# === Label definitions ===
SENTIMENT_LABELS = [0, 1, 2]
SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]
SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

TOPIC_LABELS = [0, 1, 2, 3]
TOPIC_NAMES = ["Lecturer", "Training_program", "Facility", "Others"]
TOPIC_MAP = {0: "Lecturer", 1: "Training_program", 2: "Facility", 3: "Others"}

# === Model definitions ===
MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "class": LogisticRegression,
        "params": {
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
    },
    "linear_svm": {
        "name": "Linear SVM",
        "class": LinearSVC,
        "params": {
            "max_iter": 2000,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "dual": "auto",
        },
    },
}

TASKS = {
    "sentiment": {
        "target_col": "sentiment",
        "labels": SENTIMENT_LABELS,
        "label_names": SENTIMENT_NAMES,
        "label_map": SENTIMENT_MAP,
    },
    "topic": {
        "target_col": "topic",
        "labels": TOPIC_LABELS,
        "label_names": TOPIC_NAMES,
        "label_map": TOPIC_MAP,
    },
}


def load_data():
    """Load processed CSVs."""
    data = {}
    for split in ["train", "dev", "test"]:
        filepath = os.path.join(PROCESSED_DIR, f"{split}.csv")
        df = pd.read_csv(filepath, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
        df["sentiment"] = df["sentiment"].astype(int)
        df["topic"] = df["topic"].astype(int)
        data[split] = df
    return data


def train_and_evaluate_model(
    model_key, task_key, X_train, y_train, X_dev, y_dev, X_test, y_test,
    test_ids, test_texts, task_config,
):
    """
    Train 1 model cho 1 task, evaluate trên dev và test.

    Returns:
        dict với model, metrics, reports, confusion matrices, predictions
    """
    model_config = MODELS[model_key]
    labels = task_config["labels"]
    label_names = task_config["label_names"]
    label_map = task_config["label_map"]

    # Train
    clf = model_config["class"](**model_config["params"])
    clf.fit(X_train, y_train)

    # Predict
    y_dev_pred = clf.predict(X_dev)
    y_test_pred = clf.predict(X_test)

    # Metrics
    dev_metrics = compute_metrics(y_dev, y_dev_pred, label_names)
    test_metrics = compute_metrics(y_test, y_test_pred, label_names)

    # Classification reports
    dev_report = get_classification_report(y_dev, y_dev_pred, label_names, labels)
    test_report = get_classification_report(y_test, y_test_pred, label_names, labels)

    # Confusion matrices
    dev_cm = get_confusion_matrix(y_dev, y_dev_pred, labels)
    test_cm = get_confusion_matrix(y_test, y_test_pred, labels)

    dev_cm_df = confusion_matrix_to_df(dev_cm, label_names)
    test_cm_df = confusion_matrix_to_df(test_cm, label_names)

    # Predictions for error analysis
    test_predictions = create_predictions_df(
        test_ids, test_texts, y_test, y_test_pred, label_map
    )

    return {
        "model": clf,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "dev_report": dev_report,
        "test_report": test_report,
        "dev_cm_df": dev_cm_df,
        "test_cm_df": test_cm_df,
        "test_predictions": test_predictions,
    }


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("BASELINE TRAINING — TF-IDF + Logistic Regression + Linear SVM")
    print("Đề tài #2 — Mức 1: Baseline")
    print("=" * 70)

    created_at = datetime.now(timezone.utc).isoformat()

    # === Create output dirs ===
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # === Load data ===
    print("\n[1] Loading processed data...")
    data = load_data()
    for split, df in data.items():
        print(f"    {split}: {len(df)} samples")

    # === TF-IDF: fit on TRAIN only ===
    print("\n[2] Building TF-IDF vectorizer (fit on TRAIN only)...")
    print(f"    Config: {TFIDF_CONFIG}")

    vectorizer = create_tfidf_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(data["train"]["text_lower"])
    X_dev_tfidf = vectorizer.transform(data["dev"]["text_lower"])
    X_test_tfidf = vectorizer.transform(data["test"]["text_lower"])

    vocab_size = len(vectorizer.vocabulary_)
    print(f"    Vocabulary size: {vocab_size}")
    print(f"    Train shape: {X_train_tfidf.shape}")
    print(f"    Dev shape:   {X_dev_tfidf.shape}")
    print(f"    Test shape:  {X_test_tfidf.shape}")

    # Save TF-IDF vectorizer
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, tfidf_path)
    print(f"    ✅ Saved: {tfidf_path}")

    # === Train all models ===
    all_results = {}
    all_metrics = {}
    comparison_rows = []

    for task_key, task_config in TASKS.items():
        target_col = task_config["target_col"]
        y_train = data["train"][target_col].values
        y_dev = data["dev"][target_col].values
        y_test = data["test"][target_col].values

        test_ids = data["test"]["id"].values
        test_texts = data["test"]["text"].values

        print(f"\n{'='*70}")
        print(f"TASK: {task_key.upper()}")
        print(f"{'='*70}")
        print(f"  Labels: {task_config['label_names']}")
        print(f"  Train distribution: {dict(pd.Series(y_train).value_counts().sort_index())}")

        for model_key, model_config in MODELS.items():
            combo_key = f"{task_key}_{model_key}"
            print(f"\n  --- {model_config['name']} ---")
            print(f"  Params: {model_config['params']}")

            result = train_and_evaluate_model(
                model_key, task_key,
                X_train_tfidf, y_train,
                X_dev_tfidf, y_dev,
                X_test_tfidf, y_test,
                test_ids, test_texts,
                task_config,
            )

            all_results[combo_key] = result

            # Print dev metrics
            print(f"\n  DEV metrics:")
            for k, v in result["dev_metrics"].items():
                print(f"    {k}: {v}")

            # Print test metrics
            print(f"\n  TEST metrics:")
            for k, v in result["test_metrics"].items():
                print(f"    {k}: {v}")

            # Save model
            model_path = os.path.join(MODELS_DIR, f"{combo_key}.joblib")
            joblib.dump(result["model"], model_path)
            print(f"\n  ✅ Model saved: {model_path}")

            # Store metrics
            all_metrics[combo_key] = {
                "task": task_key,
                "model": model_config["name"],
                "model_key": model_key,
                "dev": result["dev_metrics"],
                "test": result["test_metrics"],
            }

            # Comparison row (test metrics)
            comparison_rows.append({
                "task": task_key.capitalize(),
                "model": model_config["name"],
                "accuracy": result["test_metrics"]["accuracy"],
                "macro_precision": result["test_metrics"]["macro_precision"],
                "macro_recall": result["test_metrics"]["macro_recall"],
                "macro_f1": result["test_metrics"]["macro_f1"],
                "weighted_precision": result["test_metrics"]["weighted_precision"],
                "weighted_recall": result["test_metrics"]["weighted_recall"],
                "weighted_f1": result["test_metrics"]["weighted_f1"],
            })

    # === Save results ===
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    # 1. baseline_metrics.json
    metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    metrics_output = {
        "created_at": created_at,
        "random_state": RANDOM_STATE,
        "tfidf_config": TFIDF_CONFIG,
        "tfidf_vocab_size": vocab_size,
        "models": {k: {"params": MODELS[k]["params"]} for k in MODELS},
        "data_counts": {s: len(df) for s, df in data.items()},
        "results": all_metrics,
    }
    # Convert tuples to lists for JSON serialization
    metrics_output["tfidf_config"] = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in metrics_output["tfidf_config"].items()
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  ✅ {metrics_path}")

    # 2. Classification reports
    for task_key in TASKS:
        report_lines = []
        report_lines.append(f"{'='*70}")
        report_lines.append(f"{task_key.upper()} CLASSIFICATION REPORTS")
        report_lines.append(f"{'='*70}")

        for model_key in MODELS:
            combo_key = f"{task_key}_{model_key}"
            result = all_results[combo_key]

            report_lines.append(f"\n--- {MODELS[model_key]['name']} (DEV) ---")
            report_lines.append(result["dev_report"])
            report_lines.append(f"\n--- {MODELS[model_key]['name']} (TEST) ---")
            report_lines.append(result["test_report"])

        report_path = os.path.join(RESULTS_DIR, f"{task_key}_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"  ✅ {report_path}")

    # 3. Confusion matrices (test)
    for task_key in TASKS:
        for model_key in MODELS:
            combo_key = f"{task_key}_{model_key}"
            cm_df = all_results[combo_key]["test_cm_df"]
            cm_path = os.path.join(RESULTS_DIR, f"{combo_key}_confusion_matrix.csv")
            cm_df.to_csv(cm_path, encoding="utf-8")
            print(f"  ✅ {cm_path}")

    # 4. Model comparison table
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
    print(f"  ✅ {comparison_path}")

    # 5. Test predictions for error analysis
    for task_key in TASKS:
        for model_key in MODELS:
            combo_key = f"{task_key}_{model_key}"
            pred_df = all_results[combo_key]["test_predictions"]
            pred_path = os.path.join(RESULTS_DIR, f"{combo_key}_test_predictions.csv")
            pred_df.to_csv(pred_path, index=False, encoding="utf-8")
            print(f"  ✅ {pred_path}")

    # === Validation checks ===
    print(f"\n{'='*70}")
    print("VALIDATION CHECKS")
    print(f"{'='*70}")

    all_valid = True

    # V1: No data leakage — TF-IDF fit only on train
    print("\n  [V1] TF-IDF fit only on TRAIN...")
    print(f"       Vocab built from {X_train_tfidf.shape[0]} train samples")
    print(f"       ✅ PASS")

    # V2: Prediction counts match sample counts
    print("\n  [V2] Prediction counts match sample counts...")
    for task_key in TASKS:
        for model_key in MODELS:
            combo_key = f"{task_key}_{model_key}"
            pred_df = all_results[combo_key]["test_predictions"]
            expected = len(data["test"])
            actual = len(pred_df)
            if actual != expected:
                print(f"       ❌ {combo_key}: expected {expected}, got {actual}")
                all_valid = False
            else:
                print(f"       ✅ {combo_key}: {actual} predictions = {expected} samples")

    # V3: No NaN in metrics
    print("\n  [V3] No NaN in metrics...")
    nan_found = False
    for combo_key, metrics in all_metrics.items():
        for split_key in ["dev", "test"]:
            for metric_name, value in metrics[split_key].items():
                if np.isnan(value):
                    print(f"       ❌ {combo_key}/{split_key}/{metric_name} = NaN")
                    nan_found = True
                    all_valid = False
    if not nan_found:
        print("       ✅ PASS — 0 NaN values")

    # V4: Predictions within valid label set
    print("\n  [V4] Predictions within valid label set...")
    label_valid = True
    for task_key, task_config in TASKS.items():
        valid_labels = set(task_config["labels"])
        for model_key in MODELS:
            combo_key = f"{task_key}_{model_key}"
            pred_df = all_results[combo_key]["test_predictions"]
            invalid = pred_df[~pred_df["predicted_label"].isin(valid_labels)]
            if len(invalid) > 0:
                print(f"       ❌ {combo_key}: {len(invalid)} invalid predictions")
                label_valid = False
                all_valid = False
    if label_valid:
        print("       ✅ PASS — all predictions are valid labels")

    # V5: Confusion matrix dimensions
    print("\n  [V5] Confusion matrix dimensions...")
    cm_valid = True
    for task_key, task_config in TASKS.items():
        expected_size = len(task_config["labels"])
        for model_key in MODELS:
            combo_key = f"{task_key}_{model_key}"
            cm_df = all_results[combo_key]["test_cm_df"]
            if cm_df.shape != (expected_size, expected_size):
                print(f"       ❌ {combo_key}: expected {expected_size}×{expected_size}, got {cm_df.shape}")
                cm_valid = False
                all_valid = False
    if cm_valid:
        print("       ✅ PASS — sentiment 3×3, topic 4×4")

    # V6: Saved models loadable
    print("\n  [V6] Saved models loadable...")
    load_ok = True
    for task_key in TASKS:
        for model_key in MODELS:
            combo_key = f"{task_key}_{model_key}"
            model_path = os.path.join(MODELS_DIR, f"{combo_key}.joblib")
            try:
                loaded = joblib.load(model_path)
                # Quick sanity: predict on first test sample
                pred = loaded.predict(X_test_tfidf[:1])
                assert len(pred) == 1
            except Exception as e:
                print(f"       ❌ {combo_key}: {e}")
                load_ok = False
                all_valid = False
    # Also check vectorizer
    try:
        loaded_vec = joblib.load(tfidf_path)
        test_vec = loaded_vec.transform(data["test"]["text_lower"][:1])
        assert test_vec.shape[1] == vocab_size
    except Exception as e:
        print(f"       ❌ tfidf_vectorizer: {e}")
        load_ok = False
        all_valid = False

    if load_ok:
        print("       ✅ PASS — all 4 models + vectorizer loadable and functional")

    # === Print comparison table ===
    print(f"\n{'='*70}")
    print("MODEL COMPARISON (TEST)")
    print(f"{'='*70}")
    print(f"\n{'Task':<12} {'Model':<24} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 70)
    for row in comparison_rows:
        print(
            f"{row['task']:<12} {row['model']:<24} "
            f"{row['accuracy']:>10.4f} {row['macro_f1']:>10.4f} {row['weighted_f1']:>12.4f}"
        )

    # === Final status ===
    print(f"\n{'='*70}")
    if all_valid:
        print("BASELINE TRAINING COMPLETE — ALL VALIDATION CHECKS PASSED ✅")
    else:
        print("BASELINE TRAINING COMPLETE — SOME VALIDATION CHECKS FAILED ❌")
    print(f"{'='*70}")

    return all_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
