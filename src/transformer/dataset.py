"""Dataset for PhoBERT — reads processed CSV, tokenizes with PhoBERT tokenizer."""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["sentiment"] = df["sentiment"].astype(int)
    df["topic"] = df["topic"].astype(int)
    return df


def load_class_weights(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Keys are "sentiment_weights"/"topic_weights", sub-keys are strings
    result = {}
    for key, weights in data.items():
        if key.endswith("_weights") and isinstance(weights, dict):
            task = key.replace("_weights", "")
            result[task] = {int(k): v for k, v in weights.items()}
    return result


class FeedbackDataset(Dataset):
    """Generic dataset for sentiment or topic classification.

    text_col: column to use as input (text_clean for PhoBERT).
    label_col: 'sentiment' or 'topic'.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str = "text_clean",
        label_col: str = "sentiment",
        max_length: int = 128,
    ):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
        self.ids = df["id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx]) if self.texts[idx] else ""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "id": self.ids[idx],
            "text": text,
        }


def token_length_stats(
    df: pd.DataFrame,
    tokenizer,
    text_col: str = "text_clean",
    sample_size: int = None,
) -> dict:
    """Compute token-length distribution for a split."""
    import numpy as np

    texts = df[text_col].tolist()
    if sample_size and len(texts) > sample_size:
        texts = texts[:sample_size]

    lengths = []
    for t in texts:
        t = str(t) if t else ""
        lengths.append(len(tokenizer(t)["input_ids"]))

    arr = np.array(lengths)
    return {
        "count": len(arr),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 2),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "truncated_at_128": int((arr > 128).sum()),
        "truncated_at_128_pct": round((arr > 128).mean() * 100, 2),
        "truncated_at_96": int((arr > 96).sum()),
        "truncated_at_96_pct": round((arr > 96).mean() * 100, 2),
        "truncated_at_64": int((arr > 64).sum()),
        "truncated_at_64_pct": round((arr > 64).mean() * 100, 2),
    }
