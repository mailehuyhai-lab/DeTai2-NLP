"""Tab 3 — Sentiment × Topic."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import loader


def _clean_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cột đầu là nhãn sentiment (index), các cột còn lại là topic."""
    d = df.copy()
    d = d.set_index(d.columns[0])
    d.index.name = None
    return d


def render() -> None:
    st.header("Sentiment × Topic")
    counts = _clean_matrix(loader.load_eda_table("sentiment_topic_counts"))
    row_pct = _clean_matrix(loader.load_eda_table("sentiment_topic_row_pct"))

    st.subheader("Số lượng")
    st.dataframe(counts, use_container_width=True)
    st.bar_chart(counts)

    st.subheader("Tỉ lệ theo từng sentiment (%)")
    st.dataframe(row_pct.style.format("{:.1f}"), use_container_width=True)
    st.bar_chart(row_pct)

    st.info(
        "Mô tả từ EDA: phản hồi Positive tập trung chủ yếu ở Lecturer; "
        "Neutral có tỉ trọng Others tương đối cao; Facility hầu như không có "
        "phản hồi Positive. Đây là các mô tả thống kê, không phải kết luận nhân quả."
    )
