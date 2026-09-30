"""Streamlit entry point for the analytical MỨC 1 dashboard.

Launch from the project root:
    python -m streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make imports reliable both with ``streamlit run`` and module execution.
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

from src.dashboard.views import (
    batch,
    crosstab,
    error_low_confidence,
    errors,
    keywords,
    labels,
    length,
    model_info,
    models,
    overview,
    prediction,
    time_notice,
)


st.set_page_config(
    page_title="Dashboard MỨC 1 — UIT-VSFC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Dashboard phân tích phản hồi người dùng")
st.caption("UIT-VSFC · NLP sentiment và topic classification · Phạm vi MỨC 1")

with st.sidebar:
    st.header("Điều hướng")
    page = st.radio(
        "Chọn nội dung",
        [
            "Tổng quan",
            "Phân bố nhãn",
            "Sentiment × Topic",
            "Từ khóa & Emoji",
            "Độ dài văn bản",
            "So sánh mô hình",
            "Phân tích lỗi",
            "Demo dự đoán",
            "Batch Analysis",
            "Model Info",
            "Error / Low Confidence",
            "Thời gian",
        ],
    )
    st.divider()
    st.caption("Ontology / KG / RAG / rule fusion chưa được triển khai.")

try:
    if page == "Tổng quan":
        overview.render()
    elif page == "Phân bố nhãn":
        labels.render()
    elif page == "Sentiment × Topic":
        crosstab.render()
    elif page == "Từ khóa & Emoji":
        keywords.render()
    elif page == "Độ dài văn bản":
        length.render()
    elif page == "So sánh mô hình":
        models.render()
    elif page == "Phân tích lỗi":
        errors.render()
    elif page == "Demo dự đoán":
        prediction.render()
    elif page == "Batch Analysis":
        batch.render()
    elif page == "Model Info":
        model_info.render()
    elif page == "Error / Low Confidence":
        error_low_confidence.render()
    else:
        time_notice.render()
except FileNotFoundError as exc:
    st.error("Không thể tải dữ liệu dashboard.")
    st.code(str(exc))
except (KeyError, ValueError) as exc:
    st.error("Artifact dashboard có cấu trúc không đúng hoặc không đầy đủ.")
    st.code(str(exc))
