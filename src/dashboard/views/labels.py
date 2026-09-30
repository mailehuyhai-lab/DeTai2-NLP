"""Tab 2 — Phân bố nhãn."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import loader

_SPLIT_LABELS = {"train": "Train", "dev": "Dev", "test": "Test"}


def _distribution_table(dist: pd.DataFrame, split: str, task: str) -> pd.DataFrame:
    """Từ split_distribution_summary.csv trích bảng cho 1 split + 1 task."""
    d = dist[(dist["split"] == split) & (dist["task"] == task)].copy()
    d = d[["label_name", "count", "percentage"]]
    d.columns = ["Nhãn", "Số mẫu", "Tỉ lệ (%)"]
    return d.reset_index(drop=True)


def render() -> None:
    st.header("Phân bố nhãn")

    dist = loader.load_eda_table("split_distribution_summary")

    split_label = st.radio(
        "Tập dữ liệu",
        options=list(_SPLIT_LABELS.values()),
        horizontal=True,
    )
    split = {v: k for k, v in _SPLIT_LABELS.items()}[split_label]

    col_s, col_t = st.columns(2)

    for col, task, title in (
        (col_s, "sentiment", "Sentiment"),
        (col_t, "topic", "Topic"),
    ):
        with col:
            st.subheader(title)
            table = _distribution_table(dist, split, task)
            st.dataframe(table, use_container_width=True, hide_index=True)
            chart = table.set_index("Nhãn")[["Số mẫu"]]
            st.bar_chart(chart)

    st.caption(
        "Nguồn: results/eda/tables/split_distribution_summary.csv "
        "(đã được kiểm chứng qua EDA validation)."
    )
