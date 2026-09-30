"""Train PhoBERT for one task (sentiment or topic).

Usage: python src/transformer/train_task.py --task sentiment
       python src/transformer/train_task.py --task topic

Model selection: best DEV macro F1 across epochs.
TEST evaluated only once, after best checkpoint is selected.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.transformer.config import (
    BATCH_SIZE,
    CLASS_WEIGHTS_JSON,
    DATA_DIR,
    EPOCHS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MODEL_NAME,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    SENTIMENT_MAP,
    SENTIMENT_NUM_LABELS,
    TOPIC_MAP,
    TOPIC_NUM_LABELS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.transformer.dataset import FeedbackDataset, load_class_weights, load_split
from src.transformer.metrics import (
    compute_metrics,
    confusion_matrix_df,
    per_class_report,
    predictions_df,
)
from src.transformer.model import WeightedLossTrainer, build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "id": [b["id"] for b in batch],
        "text": [b["text"] for b in batch],
    }


def run(task: str, max_length: int) -> dict:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Task: {task}  |  Device: {device}  |  Seed: {SEED}")
    print(f"Model: {MODEL_NAME}  |  max_length: {max_length}")
    print(f"{'='*60}")

    label_map = SENTIMENT_MAP if task == "sentiment" else TOPIC_MAP
    num_labels = SENTIMENT_NUM_LABELS if task == "sentiment" else TOPIC_NUM_LABELS
    label_names = [label_map[i] for i in range(num_labels)]

    # Data
    train_df = load_split(DATA_DIR / "train.csv")
    dev_df = load_split(DATA_DIR / "dev.csv")
    test_df = load_split(DATA_DIR / "test.csv")
    print(f"Train={len(train_df)}  Dev={len(dev_df)}  Test={len(test_df)}", flush=True)

    # Class weights from preprocessing step
    all_weights = load_class_weights(CLASS_WEIGHTS_JSON)
    task_weights = all_weights[task]
    weight_tensor = torch.tensor(
        [task_weights[i] for i in range(num_labels)], dtype=torch.float32
    )
    print(f"Class weights: {dict(zip(label_names, weight_tensor.tolist()))}", flush=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets & loaders
    train_ds = FeedbackDataset(train_df, tokenizer, "text_clean", task, max_length)
    dev_ds = FeedbackDataset(dev_df, tokenizer, "text_clean", task, max_length)
    test_ds = FeedbackDataset(test_df, tokenizer, "text_clean", task, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        collate_fn=collate_fn,
    )

    # Model — full fine-tuning (all layers trainable)
    model = build_model(MODEL_NAME, num_labels)
    trainer = WeightedLossTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        epochs=EPOCHS,
        class_weights=weight_tensor,
        gradient_clip=GRADIENT_CLIP,
    )

    total_steps = len(train_loader) * EPOCHS
    trainer.setup_scheduler(total_steps)
    print(f"Total steps: {total_steps}  (batches/epoch: {len(train_loader)})", flush=True)

    # Output dirs
    model_dir = MODELS_DIR / task
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training loop — best DEV macro F1 checkpoint
    best_dev_f1 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        log_path = str(results_dir / f"{task}_train.log")
        train_stats = trainer.train_epoch(train_loader, log_path=log_path)

        dev_res = trainer.predict(dev_loader)
        dev_metrics = compute_metrics(dev_res["labels"], dev_res["predictions"], label_names)

        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "time_s": round(elapsed, 1),
        }
        history.append(row)
        print(f"Epoch {epoch}: loss={row['train_loss']:.4f}  "
              f"dev_acc={row['dev_accuracy']:.4f}  "
              f"dev_macroF1={row['dev_macro_f1']:.4f}  "
              f"dev_wF1={row['dev_weighted_f1']:.4f}  "
              f"({elapsed:.0f}s)", flush=True)

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            # Save best checkpoint
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            # Save label mapping
            with open(model_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)
            print(f"  -> New best (macro F1={best_dev_f1:.4f}), checkpoint saved", flush=True)

    print(f"\nBest epoch: {best_epoch} (dev macro F1 = {best_dev_f1:.4f})", flush=True)

    # ---- Final evaluation on TEST with best checkpoint ----
    print("\n--- Evaluating TEST with best checkpoint ---", flush=True)
    # Reload best model
    from transformers import AutoModelForSequenceClassification

    best_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    best_model.to(device)
    trainer.model = best_model

    # Dev final (with best model for report)
    dev_res = trainer.predict(dev_loader)
    dev_metrics = compute_metrics(dev_res["labels"], dev_res["predictions"], label_names)
    dev_report = per_class_report(dev_res["labels"], dev_res["predictions"], label_names)
    dev_cm = confusion_matrix_df(dev_res["labels"], dev_res["predictions"], label_names)

    # Test final
    test_res = trainer.predict(test_loader)
    test_metrics = compute_metrics(test_res["labels"], test_res["predictions"], label_names)
    test_report = per_class_report(test_res["labels"], test_res["predictions"], label_names)
    test_cm = confusion_matrix_df(test_res["labels"], test_res["predictions"], label_names)
    test_preds_df = predictions_df(
        test_res["ids"], test_res["texts"],
        test_res["labels"], test_res["predictions"], label_map,
    )

    # Save results
    test_preds_df.to_csv(results_dir / f"{task}_test_predictions.csv", index=False)
    test_cm.to_csv(results_dir / f"{task}_confusion_matrix.csv")
    dev_cm.to_csv(results_dir / f"{task}_dev_confusion_matrix.csv")

    report_text = (
        f"=== DEV (best epoch {best_epoch}) ===\n{dev_report}\n\n"
        f"=== TEST ===\n{test_report}"
    )
    with open(results_dir / f"{task}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(results_dir / f"{task}_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    result = {
        "task": task,
        "model": MODEL_NAME,
        "seed": SEED,
        "max_length": max_length,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs_trained": EPOCHS,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_dev_f1,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "class_weights": {label_map[i]: task_weights[i] for i in range(num_labels)},
        "training_history": history,
    }
    with open(results_dir / f"{task}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nTest: acc={test_metrics['accuracy']}  "
          f"macroF1={test_metrics['macro_f1']}  "
          f"wF1={test_metrics['weighted_f1']}", flush=True)
    print(f"Artifacts: {model_dir}, {results_dir}", flush=True)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["sentiment", "topic"])
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()
    run(args.task, args.max_length)
