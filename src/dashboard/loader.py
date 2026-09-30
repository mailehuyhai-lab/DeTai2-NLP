"""Đọc các artifact hiện có của project cho dashboard.

Mọi đường dẫn đều là project-relative thông qua Path(__file__).
Không hardcode đường dẫn tuyệt đối của máy.
Mọi hàm đọc chỉ đọc; nếu thiếu artifact sẽ raise FileNotFoundError rõ ràng
thay vì âm thầm tạo dữ liệu thay thế.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# src/dashboard/loader.py -> parents[2] là project root
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
BASELINE_DIR = RESULTS_DIR / "baseline"
TRANSFORMER_DIR = RESULTS_DIR / "transformer"
EDA_DIR = RESULTS_DIR / "eda"
EDA_TABLES_DIR = EDA_DIR / "tables"
EDA_FIG_DIR = EDA_DIR / "figures"
EDA_SUM_DIR = EDA_DIR / "summaries"
ERROR_DIR = RESULTS_DIR / "error_analysis"
REPORTS_DIR = BASE_DIR / "reports"


def _need(path: Path) -> Path:
    """Trả về path nếu tồn tại, ngược lại raise lỗi có chỉ dẫn."""
    if not path.exists():
        raise FileNotFoundError(
            f"Thiếu artifact cần thiết: {path.relative_to(BASE_DIR)}\n"
            "Chạy các bước pipeline trước (preprocessing/baseline/eda/transformer)."
        )
    return path


@st.cache_data(show_spinner=False)
def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(_need(path), **kwargs)


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    with open(_need(path), encoding="utf-8") as f:
        return json.load(f)


# --- Processed data -------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_split(split: str) -> pd.DataFrame:
    """split ∈ {train, dev, test}."""
    return load_csv(DATA_DIR / f"{split}.csv")


@st.cache_data(show_spinner=False)
def load_all_splits() -> pd.DataFrame:
    frames = []
    for s in ("train", "dev", "test"):
        d = load_split(s).copy()
        d["split"] = s
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_label_maps() -> dict:
    return load_json(DATA_DIR / "label_maps.json")


@st.cache_data(show_spinner=False)
def load_class_weights() -> dict:
    return load_json(DATA_DIR / "class_weights.json")


# --- Baseline -------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_baseline_metrics() -> dict:
    return load_json(BASELINE_DIR / "baseline_metrics.json")


def baseline_pred_path(task: str, model_key: str) -> Path:
    return BASELINE_DIR / f"{task}_{model_key}_test_predictions.csv"


def baseline_cm_path(task: str, model_key: str) -> Path:
    return BASELINE_DIR / f"{task}_{model_key}_confusion_matrix.csv"


# --- Transformer ----------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_tf_metrics(task: str) -> dict:
    return load_json(TRANSFORMER_DIR / f"{task}_metrics.json")


@st.cache_data(show_spinner=False)
def load_tf_history(task: str):
    return load_json(TRANSFORMER_DIR / f"{task}_training_history.json")


def tf_pred_path(task: str) -> Path:
    return TRANSFORMER_DIR / f"{task}_test_predictions.csv"


def tf_cm_path(task: str) -> Path:
    return TRANSFORMER_DIR / f"{task}_confusion_matrix.csv"


@st.cache_data(show_spinner=False)
def load_tf_predictions(task: str) -> pd.DataFrame:
    """Load a Transformer test-prediction artifact and validate its contract."""
    if task not in {"sentiment", "topic"}:
        raise ValueError(f"Task không hợp lệ: {task}")
    frame = load_csv(tf_pred_path(task)).copy()
    required = {
        "id", "text", "true_label", "predicted_label", "correct",
        "true_label_name", "predicted_label_name",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Artifact dự đoán {task} thiếu cột bắt buộc: {', '.join(missing)}"
        )
    return frame


@st.cache_data(show_spinner=False)
def load_tf_model_info(task: str) -> dict:
    """Return the saved Transformer metrics without inventing defaults."""
    if task not in {"sentiment", "topic"}:
        raise ValueError(f"Task không hợp lệ: {task}")
    metrics = load_tf_metrics(task)
    required = {"model", "test_metrics"}
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(
            f"Metrics artifact {task} thiếu trường bắt buộc: {', '.join(missing)}"
        )
    return metrics


def transformer_prediction_has_confidence(frame: pd.DataFrame) -> bool:
    """Whether a saved prediction frame contains real confidence values."""
    confidence_columns = {
        "confidence", "prediction_confidence", "probability", "max_probability",
    }
    return bool(confidence_columns.intersection(frame.columns))

@st.cache_data(show_spinner=False)
def load_eda_table(name: str) -> pd.DataFrame:
    """name là tên file không có .csv trong results/eda/tables/."""
    return load_csv(EDA_TABLES_DIR / f"{name}.csv")


@st.cache_data(show_spinner=False)
def load_eda_summary(name: str) -> dict:
    return load_json(EDA_SUM_DIR / f"{name}.json")


def eda_figure(name: str) -> Path:
    return _need(EDA_FIG_DIR / name)


# --- Error analysis -------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_error_statistics() -> dict:
    return load_json(ERROR_DIR / "error_statistics.json")


@st.cache_data(show_spinner=False)
def load_error_summary(task: str) -> pd.DataFrame:
    return load_csv(ERROR_DIR / f"{task}_error_summary.csv")


@st.cache_data(show_spinner=False)
def load_error_samples(task: str) -> pd.DataFrame:
    return load_csv(ERROR_DIR / f"{task}_error_samples.csv")
