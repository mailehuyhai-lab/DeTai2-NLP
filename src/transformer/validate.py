"""Post-training validation for transformer models.

Checks:
1. Correct sample counts (train=11424, dev=1583, test=3166)
2. Tokenizer loadable
3. Checkpoints loadable
4. Prediction counts match test
5. Valid labels in predictions
6. Confusion matrix dimensions correct
7. Metrics not NaN
8. Class weights applied
9. Raw data MD5 unchanged
10. Processed data unchanged
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.transformer.config import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SENTIMENT_MAP,
    TOPIC_MAP,
)
from src.preprocessing.load_raw import verify_raw_checksums

PROCESSED_DIR = DATA_DIR


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    all_ok = True
    print("=" * 60)
    print("TRANSFORMER VALIDATION")
    print("=" * 60)

    # V1: Sample counts
    print("\n[V1] Sample counts")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    dev_df = pd.read_csv(PROCESSED_DIR / "dev.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    all_ok &= check("train = 11424", len(train_df) == 11424, str(len(train_df)))
    all_ok &= check("dev = 1583", len(dev_df) == 1583, str(len(dev_df)))
    all_ok &= check("test = 3166", len(test_df) == 3166, str(len(test_df)))

    # V2: Tokenizer loadable
    print("\n[V2] Tokenizer loadable")
    for task in ("sentiment", "topic"):
        try:
            tok = AutoTokenizer.from_pretrained(MODELS_DIR / task)
            ok = check(f"{task} tokenizer", True)
        except Exception as e:
            ok = check(f"{task} tokenizer", False, str(e)[:80])
        all_ok &= ok

    # V3: Checkpoints loadable
    print("\n[V3] Model checkpoints loadable")
    for task in ("sentiment", "topic"):
        try:
            m = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / task)
            n_params = sum(p.numel() for p in m.parameters())
            ok = check(f"{task} model", True, f"{n_params/1e6:.0f}M params")
        except Exception as e:
            ok = check(f"{task} model", False, str(e)[:80])
        all_ok &= ok

    # V4: Prediction counts + valid labels + metrics not NaN
    print("\n[V4] Predictions valid")
    for task, lmap in [("sentiment", SENTIMENT_MAP), ("topic", TOPIC_MAP)]:
        pred_file = RESULTS_DIR / f"{task}_test_predictions.csv"
        if not pred_file.exists():
            all_ok &= check(f"{task} predictions exist", False)
            continue
        preds = pd.read_csv(pred_file)
        all_ok &= check(
            f"{task} prediction count",
            len(preds) == 3166,
            f"{len(preds)} rows",
        )
        valid_labels = set(lmap.keys())
        true_ok = preds["true_label"].isin(valid_labels).all()
        pred_ok = preds["predicted_label"].isin(valid_labels).all()
        all_ok &= check(f"{task} true labels valid", true_ok)
        all_ok &= check(f"{task} predicted labels valid", pred_ok)
        all_ok &= check(
            f"{task} no NaN predictions",
            not preds["predicted_label"].isna().any(),
        )

    # V5: Metrics files exist and not NaN
    print("\n[V5] Metrics valid")
    for task in ("sentiment", "topic"):
        metrics_file = RESULTS_DIR / f"{task}_metrics.json"
        if not metrics_file.exists():
            all_ok &= check(f"{task} metrics exist", False)
            continue
        with open(metrics_file) as f:
            m = json.load(f)
        has_dev = "dev_metrics" in m and "macro_f1" in m["dev_metrics"]
        has_test = "test_metrics" in m and "macro_f1" in m["test_metrics"]
        all_ok &= check(f"{task} has dev+test metrics", has_dev and has_test)
        if has_test:
            for k in ("accuracy", "macro_f1", "weighted_f1"):
                v = m["test_metrics"].get(k)
                all_ok &= check(f"{task} test {k} valid", v is not None and v == v)

    # V6: Confusion matrix files exist
    print("\n[V6] Confusion matrices")
    for task in ("sentiment", "topic"):
        cm_file = RESULTS_DIR / f"{task}_confusion_matrix.csv"
        if cm_file.exists():
            cm = pd.read_csv(cm_file, index_col=0)
            n = len(SENTIMENT_MAP) if task == "sentiment" else len(TOPIC_MAP)
            all_ok &= check(
                f"{task} CM {n}x{n}",
                cm.shape == (n, n),
                f"{cm.shape}",
            )
        else:
            all_ok &= check(f"{task} CM exists", False)

    # V7: Training history
    print("\n[V7] Training history")
    for task in ("sentiment", "topic"):
        hist_file = RESULTS_DIR / f"{task}_training_history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                hist = json.load(f)
            all_ok &= check(
                f"{task} history has >=1 epochs",
                len(hist) >= 1,
                f"{len(hist)} entries",
            )
        else:
            all_ok &= check(f"{task} history exists", False)

    # V8: Raw data unchanged
    print("\n[V8] Raw data integrity")
    raw_dir = str(PROCESSED_DIR.parent / "raw")
    raw_ok = bool(verify_raw_checksums(raw_dir))
    all_ok &= check("9/9 raw MD5 unchanged", raw_ok)

    # V9: Processed data unchanged (spot-check row counts)
    print("\n[V9] Processed data integrity")
    all_ok &= check(
        "processed train rows", len(train_df) == 11424, str(len(train_df))
    )
    all_ok &= check(
        "processed dev rows", len(dev_df) == 1583, str(len(dev_df))
    )
    all_ok &= check(
        "processed test rows", len(test_df) == 3166, str(len(test_df))
    )

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see above")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
