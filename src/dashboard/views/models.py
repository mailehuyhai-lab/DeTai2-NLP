"""Tab 6 — So sánh mô hình."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import loader


def _cm_table(path, task: str) -> pd.DataFrame:
    df = loader.load_csv(path)
    first = df.columns[0]
    if first.startswith("Unnamed"):
        df = df.set_index(first)
    df.index.name = "True\\Pred"
    return df


def render() -> None:
    st.header("So sánh mô hình")

    st.subheader("Metrics kiểm chứng trên test")
    comparison = loader.load_model_comparison()
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown(
        "**Sentiment**: PhoBERT Accuracy và Macro F1 đều cao hơn Linear SVM.\n\n"
        "**Topic**: PhoBERT Macro F1 cao hơn nhưng Accuracy thấp hơn Linear SVM một chút.\n\n"
        "Đây là so sánh mô tả giữa các artifact đã kiểm chứng, không phải phát biểu nhân quả."
    )

    st.subheader("PhoBERT training history")
    task_hist = st.selectbox("Task", ["sentiment", "topic"], key="hist_task")
    history = loader.load_tf_history(task_hist)
    hist_df = pd.DataFrame(history)
    if not hist_df.empty:
        cols = [c for c in ["epoch", "train_loss", "dev_accuracy", "dev_macro_f1", "dev_weighted_f1"] if c in hist_df.columns]
        st.dataframe(hist_df[cols], use_container_width=True, hide_index=True)
        if {"epoch", "dev_macro_f1"}.issubset(hist_df.columns):
            st.line_chart(hist_df.set_index("epoch")["dev_macro_f1"])

    st.subheader("Confusion matrices")
    for task in ("sentiment", "topic"):
        st.markdown(f"**{task.capitalize()}**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Linear SVM (test)")
            st.dataframe(_cm_table(loader.baseline_cm_path(task, "linear_svm"), task), use_container_width=True)
        with col2:
            st.caption("PhoBERT (test)")
            st.dataframe(_cm_table(loader.tf_cm_path(task), task), use_container_width=True)
