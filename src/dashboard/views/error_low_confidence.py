"""Transformer test-error and session batch confidence review."""
from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from .. import loader


_TASKS = {"Sentiment": "sentiment", "Topic": "topic"}


def _correct_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["correct"], errors="coerce").fillna(0).eq(1)


def filter_test_errors(
    frame: pd.DataFrame,
    true_label: str = "Tất cả",
    predicted_label: str = "Tất cả",
) -> pd.DataFrame:
    """Filter only rows explicitly marked incorrect by the artifact."""
    filtered = frame.loc[~_correct_mask(frame)].copy()
    if true_label != "Tất cả":
        filtered = filtered[filtered["true_label_name"] == true_label]
    if predicted_label != "Tất cả":
        filtered = filtered[filtered["predicted_label_name"] == predicted_label]
    columns = [
        "id", "text", "true_label_name", "predicted_label_name", "correct",
    ]
    return filtered[columns].reset_index(drop=True)


def filter_batch_results(
    frame: pd.DataFrame,
    task: str,
    threshold: float,
    mode: str,
) -> pd.DataFrame:
    """Filter successful session-batch rows using actual softmax values."""
    confidence_column = f"{task}_confidence"
    label_column = f"predicted_{task}"
    filtered = frame[frame["prediction_status"] == "success"].copy()
    filtered = filtered[filtered[confidence_column].notna()]
    low = pd.to_numeric(filtered[confidence_column], errors="coerce") < threshold
    if mode == "Low confidence":
        filtered = filtered[low]
    elif mode == "Low confidence + errors":
        # Batch uploads have no true labels by contract; this mode is only
        # meaningful when an optional true-label column is present.
        true_column = "true_sentiment" if task == "sentiment" else "true_topic"
        if true_column not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[low & filtered[true_column].ne(filtered[label_column])]
    elif mode == "Errors only":
        true_column = "true_sentiment" if task == "sentiment" else "true_topic"
        if true_column not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[filtered[true_column].ne(filtered[label_column])]
    return filtered.reset_index(drop=True)


def _csv_download(frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        "Download CSV",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def render() -> None:
    st.header("Error / Low Confidence")
    st.caption(
        "Lọc lỗi test từ artifact Transformer và xem các dòng confidence thấp "
        "của Batch Analysis hiện tại."
    )

    task_label = st.radio("Task test", list(_TASKS), horizontal=True, key="review_task")
    task = _TASKS[task_label]
    predictions = loader.load_tf_predictions(task)
    labels = sorted(set(predictions["true_label_name"]) | set(predictions["predicted_label_name"]))
    c1, c2 = st.columns(2)
    with c1:
        true_label = st.selectbox("True label", ["Tất cả", *labels], key="review_true")
    with c2:
        predicted_label = st.selectbox("Predicted label", ["Tất cả", *labels], key="review_pred")
    errors = filter_test_errors(predictions, true_label, predicted_label)
    st.subheader("Errors trên test artifact")
    st.metric("Số lỗi sau bộ lọc", f"{len(errors):,}")
    st.dataframe(errors, use_container_width=True, hide_index=True)
    _csv_download(errors, f"{task}_test_errors.csv", "download_test_errors")
    if not loader.transformer_prediction_has_confidence(predictions):
        st.warning(
            "Artifact test không chứa logits/xác suất/confidence. Vì vậy dashboard "
            "chỉ lọc được lỗi theo true/predicted label, không thể xác định low "
            "confidence test một cách trung thực."
        )

    st.divider()
    st.subheader("Low confidence từ Batch Analysis hiện tại")
    batch = st.session_state.get("m1_batch_results")
    if batch is None:
        st.info("Chạy Batch Analysis trước để lọc confidence softmax.")
        return

    batch_task_label = st.radio("Task batch", list(_TASKS), horizontal=True, key="batch_review_task")
    batch_task = _TASKS[batch_task_label]
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.05, key="batch_threshold")
    mode = st.selectbox(
        "Kiểu lọc",
        ["Low confidence", "Errors only", "Low confidence + errors"],
        key="batch_filter_mode",
    )
    filtered = filter_batch_results(batch, batch_task, threshold, mode)
    invalid_count = int((batch["prediction_status"] != "success").sum())
    st.caption(
        f"Đã loại {invalid_count:,} dòng input không hợp lệ. Confidence là softmax "
        "của model, chưa qua calibration."
    )
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    _csv_download(filtered, "batch_low_confidence.csv", "download_batch_review")
