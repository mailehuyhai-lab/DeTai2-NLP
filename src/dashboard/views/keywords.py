"""Tab 4 — Từ khóa & Emoji."""
from __future__ import annotations

import streamlit as st

from .. import loader


def render() -> None:
    st.header("Từ khóa & Emoji")

    st.subheader("Top keywords toàn bộ corpus")
    overall = loader.load_eda_table("keyword_frequency")
    st.dataframe(overall, use_container_width=True, hide_index=True)
    st.bar_chart(overall.head(20).set_index("keyword")["count"])

    by_class = loader.load_eda_table("keyword_frequency_by_class")
    st.subheader("Keyword theo lớp")
    task_label = st.radio("Loại nhãn", ["Sentiment", "Topic"], horizontal=True)
    task = task_label.lower()
    filtered = by_class[by_class["task"] == task]
    classes = filtered["class"].drop_duplicates().tolist()
    selected = st.selectbox("Lớp", classes)
    class_df = filtered[filtered["class"] == selected].head(20)
    st.dataframe(class_df[["keyword", "count"]], use_container_width=True, hide_index=True)
    st.bar_chart(class_df.set_index("keyword")["count"])

    st.subheader("Emoji")
    emoji = loader.load_eda_summary("emoji_summary")
    rows = []
    for split, value in emoji.items():
        rows.append({
            "Split": split,
            "Số mẫu": value["n_samples"],
            "Mẫu có emoji": value["n_with_emoji"],
            "Tỉ lệ (%)": value["pct_with_emoji"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    split = st.selectbox("Xem emoji phổ biến", list(emoji), key="emoji_split")
    top = emoji[split]["top_emojis"]
    if top:
        st.dataframe(top, use_container_width=True, hide_index=True)
    st.caption("Từ khóa được đọc từ EDA artifact; dashboard không chạy lại thuật toán trích xuất.")
