"""Inference PhoBERT cho tab Demo dự đoán (MỨC 1).

Load model/tokenizer một lần qua st.cache_resource (lazy — chỉ khi predict).
Preprocess bằng clean_text() từ src/preprocessing — đúng hàm đã sinh text_clean.
Không ghi file; chỉ đọc checkpoint trong models/transformer/.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import streamlit as st

from src.preprocessing.text_cleaner import clean_text
from src.transformer.config import MAX_LENGTH, MODELS_DIR

REQUIRED_FILES = ("model.safetensors", "config.json", "label_map.json",
                  "tokenizer_config.json")


def _task_dir(task: str) -> Path:
    return MODELS_DIR / task


def models_available() -> dict:
    """Kiểm tra artifact từng task. Trả {task: (ok, missing_files)}."""
    status = {}
    for task in ("sentiment", "topic"):
        d = _task_dir(task)
        missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
        status[task] = (not missing, missing)
    return status


@st.cache_resource(show_spinner=False)
def get_model_bundle(task: str):
    """Load tokenizer + model + label_map một lần cho mỗi task.

    Returns (tokenizer, model, label_map). Raise nếu thiếu artifact.
    """
    d = _task_dir(task)
    if not d.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {d}")
    for f in REQUIRED_FILES:
        if not (d / f).exists():
            raise FileNotFoundError(f"Checkpoint thiếu file {f}: {d}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d)
    model.to(torch.device("cpu"))
    model.eval()

    with open(d / "label_map.json", encoding="utf-8") as fh:
        label_map = {int(k): v for k, v in json.load(fh).items()}

    return tokenizer, model, label_map


def _predict_one(task: str, text_clean: str) -> dict:
    """Chạy 1 model trên text_clean; trả label + softmax confidence."""
    tokenizer, model, label_map = get_model_bundle(task)

    enc = tokenizer(
        text_clean,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # Compare with an untruncated encoding so exactly-32-token inputs are not
    # incorrectly reported as truncated.
    untruncated_len = len(tokenizer(text_clean, add_special_tokens=True)["input_ids"])
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(logits.argmax(dim=-1).item())

    return {
        "id": pred_id,
        "label": label_map[pred_id],
        "confidence": float(probs[pred_id].item()),
        "probs": {label_map[i]: float(probs[i].item()) for i in range(len(label_map))},
        "truncated": untruncated_len > MAX_LENGTH,
    }


def predict_feedback(raw_text: str) -> dict:
    """Pipeline đầy đủ: clean_text → sentiment + topic.

    Returns dict theo output contract trong DASHBOARD_PHASE2_PLAN.md.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Feedback không được rỗng hoặc chỉ chứa whitespace.")
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise ValueError("Feedback rỗng sau tiền xử lý.")
    sent = _predict_one("sentiment", cleaned)
    topic = _predict_one("topic", cleaned)
    return {
        "text_clean": cleaned,
        "truncated": sent["truncated"] or topic["truncated"],
        "sentiment": {k: sent[k] for k in ("id", "label", "confidence", "probs")},
        "topic": {k: topic[k] for k in ("id", "label", "confidence", "probs")},
    }
