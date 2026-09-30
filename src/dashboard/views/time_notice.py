"""Tab 8 — Hạn chế theo thời gian."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("Thời gian")
    st.warning("**Phân tích theo thời gian không khả dụng.**")
    st.write("UIT-VSFC không cung cấp trường timestamp trong dữ liệu hiện tại.")
    st.write(
        "Dashboard không tạo ngày giả, không suy diễn thời gian từ ID, "
        "thứ tự file hoặc bất kỳ nguồn dữ liệu thay thế nào."
    )
