"""Tab 5 — Độ dài văn bản."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import loader


def render() -> None:
    st.header("Độ dài văn bản")
    st.caption("The current Transformer configuration uses `max_length=32`.")

    summary = loader.load_eda_table("text_length_summary")
    rename = {
        "split": "Split",
        "n_samples": "Số mẫu",
        "chars_mean": "Ký tự (TB)",
        "chars_median": "Ký tự (Trung vị)",
        "chars_p95": "Ký tự (P95)",
        "words_mean": "Từ (TB)",
        "words_median": "Từ (Trung vị)",
        "phobert_tokens_mean": "PhoBERT tokens (TB)",
        "phobert_tokens_p95": "PhoBERT tokens (P95)",
        "truncated_gt_32": "Mẫu >32 tokens",
        "truncated_gt_32_pct": "Tỉ lệ >32 (%)",
    }
    cols = [c for c in rename if c in summary.columns]
    table = summary[cols].rename(columns=rename)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Phân bố độ dài")
    st.image(
        str(loader.eda_figure("text_length_distribution.png")),
        caption="Phân bố độ dài văn bản (nguồn: results/eda/figures).",
    )

    st.subheader("Truncation tại max_length=32")
    chart = summary[["split", "truncated_gt_32_pct"]].copy()
    chart.columns = ["Split", "Tỉ lệ bị truncate (%)"]
    st.bar_chart(chart.set_index("Split"))
