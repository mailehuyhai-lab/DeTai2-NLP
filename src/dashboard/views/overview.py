"""Tab 1 — Tổng quan."""
from __future__ import annotations

import streamlit as st

from .. import loader


def render() -> None:
    st.header("Tổng quan")

    st.info("**MỨC 1** — Ontology / KG / RAG / rule fusion chưa được triển khai.")

    counts = loader.load_baseline_metrics()["data_counts"]
    total = counts["train"] + counts["dev"] + counts["test"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số mẫu", f"{total:,}")
    c2.metric("Train", f"{counts['train']:,}")
    c3.metric("Dev", f"{counts['dev']:,}")
    c4.metric("Test", f"{counts['test']:,}")

    st.subheader("Pipeline hiện tại")
    st.code(
        "Raw UIT-VSFC → Preprocessing → Validation → TF-IDF Baseline "
        "→ Error Analysis → EDA → PhoBERT Transformer → Evaluation",
        language="text",
    )

    label_maps = loader.load_label_maps()
    st.subheader("Nhãn")
    col_a, col_b = st.columns(2)
    col_a.markdown("**Sentiment**")
    col_a.write(", ".join(label_maps["sentiment"].values()))
    col_b.markdown("**Topic**")
    col_b.write(", ".join(label_maps["topic"].values()))

    st.subheader("Mất cân bằng lớp (train)")
    imb = loader.load_eda_summary("imbalance_summary")
    for task, title in (("sentiment", "Sentiment"), ("topic", "Topic")):
        t = imb[task]["train"]
        st.markdown(
            f"**{title}** — đa số: `{t['majority_class']}` "
            f"({t['majority_count']:,}), thiểu số: `{t['minority_class']}` "
            f"({t['minority_count']:,}), tỉ lệ **{t['ratio_majority_to_minority']:.1f}×**"
        )

    cw = loader.load_class_weights()
    st.caption(
        "Class weights (inverse_frequency, chỉ tính trên train): "
        + cw.get("note", "")
    )
