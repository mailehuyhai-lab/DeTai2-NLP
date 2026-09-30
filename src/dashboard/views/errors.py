"""Tab 7 — Phân tích lỗi từ artifact đã có."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import loader


def _display_task(task: str, stats: dict) -> None:
    value = stats[task]
    title = "Sentiment" if task == "sentiment" else "Topic"
    st.subheader(f"{title} error analysis")

    error_rate = 100 * value["n_errors"] / value["total_test"]
    a, b, c = st.columns(3)
    a.metric("Tổng mẫu test", f"{value['total_test']:,}")
    b.metric("Số lỗi", f"{value['n_errors']:,}")
    c.metric("Tỉ lệ lỗi", f"{error_rate:.2f}%")

    st.markdown("**Top cặp nhãn bị nhầm lẫn**")
    summary = loader.load_error_summary(task)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    length = value.get("length_stats", {})
    if length:
        st.markdown("**So sánh độ dài lỗi / dự đoán đúng**")
        length_df = pd.DataFrame([
            {"Nhóm": "Dự đoán lỗi", "Số từ trung bình": length.get("error_mean_words"),
             "Trung vị": length.get("error_median_words"),
             "≤5 từ (%)": length.get("error_short_5w_pct")},
            {"Nhóm": "Dự đoán đúng", "Số từ trung bình": length.get("correct_mean_words"),
             "Trung vị": length.get("correct_median_words"), "≤5 từ (%)": length.get("correct_short_5w_pct")},
        ])
        st.dataframe(length_df, use_container_width=True, hide_index=True)

    st.markdown("**Mẫu lỗi đại diện**")
    samples = loader.load_error_samples(task).copy()
    samples.columns = [
        "ID", "Văn bản", "Nhãn đúng", "Nhãn dự đoán"
    ]
    st.dataframe(samples, use_container_width=True, hide_index=True, height=360)


def render() -> None:
    st.header("Phân tích lỗi")
    stats = loader.load_error_statistics()
    task_label = st.radio("Bài toán", ["Sentiment", "Topic"], horizontal=True)
    _display_task(task_label.lower(), stats)
    st.caption("Dashboard chỉ hiển thị artifact từ results/error_analysis/; không chạy lại error-analysis pipeline.")
