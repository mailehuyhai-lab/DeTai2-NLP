"""Model metadata and test metrics view."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import loader


_TASKS = {
    "Sentiment": ("sentiment", ["Negative", "Neutral", "Positive"]),
    "Topic": ("topic", ["Lecturer", "Training_program", "Facility", "Others"]),
}


def _display_value(mapping: dict, key: str) -> str:
    """Format metadata as text so Streamlit Arrow sees one column type."""
    value = mapping.get(key)
    return "Không có trong artifact" if value is None else str(value)


def _format_metric(value) -> str:
    return "Không có trong artifact" if value is None else f"{value:.4f}"


def _render_task(task: str, classes: list[str]) -> None:
    metrics = loader.load_tf_model_info(task)
    test = metrics.get("test_metrics", {})

    st.subheader(f"{task.capitalize()} Classification")
    config = pd.DataFrame([
        {"Thuộc tính": "Model", "Giá trị": _display_value(metrics, "model")},
        {"Thuộc tính": "Task", "Giá trị": "Sentiment Classification" if task == "sentiment" else "Topic Classification"},
        {"Thuộc tính": "Classes", "Giá trị": ", ".join(classes)},
        {"Thuộc tính": "Seed", "Giá trị": _display_value(metrics, "seed")},
        {"Thuộc tính": "max_length", "Giá trị": _display_value(metrics, "max_length")},
        {"Thuộc tính": "batch_size", "Giá trị": _display_value(metrics, "batch_size")},
        {"Thuộc tính": "learning_rate", "Giá trị": _display_value(metrics, "learning_rate")},
        {"Thuộc tính": "epochs_trained", "Giá trị": _display_value(metrics, "epochs_trained")},
        {"Thuộc tính": "best_epoch", "Giá trị": _display_value(metrics, "best_epoch")},
    ])
    st.dataframe(config, use_container_width=True, hide_index=True)

    a, b, c = st.columns(3)
    a.metric("Accuracy test", _format_metric(test.get("accuracy")))
    b.metric("Macro-F1 test", _format_metric(test.get("macro_f1")))
    c.metric("Weighted-F1 test", _format_metric(test.get("weighted_f1")))

    detail = pd.DataFrame([
        {"Metric": "Accuracy", "Test": test.get("accuracy")},
        {"Metric": "Macro Precision", "Test": test.get("macro_precision")},
        {"Metric": "Macro Recall", "Test": test.get("macro_recall")},
        {"Metric": "Macro F1", "Test": test.get("macro_f1")},
        {"Metric": "Weighted F1", "Test": test.get("weighted_f1")},
    ])
    st.dataframe(detail.style.format({"Test": "{:.4f}"}, na_rep="Không có trong artifact"), use_container_width=True, hide_index=True)


def render() -> None:
    st.header("Model Info")
    st.caption(
        "Thông tin dưới đây được đọc từ results/transformer/*_metrics.json; "
        "dashboard không huấn luyện hoặc đánh giá lại model."
    )
    task_label = st.radio("Bài toán", list(_TASKS), horizontal=True, key="model_info_task")
    task, classes = _TASKS[task_label]
    _render_task(task, classes)
    st.info(
        "Confidence trong các tab dự đoán là xác suất softmax của model, "
        "chưa qua calibration. Sentiment và Topic là hai task độc lập."
    )
