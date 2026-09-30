"""Tab Demo dự đoán — Phase 2 (MỨC 1).

Nhập feedback tiếng Việt → clean_text → PhoBERT sentiment + topic.
Hiển thị nhãn + xác suất softmax của model (không calibration).
"""
from __future__ import annotations

import streamlit as st

from .. import inference


def render() -> None:
    st.header("Demo dự đoán")
    st.caption(
        "Nhập một phản hồi tiếng Việt → tiền xử lý `clean_text` → "
        "PhoBERT dự đoán Sentiment và Topic (MỨC 1)."
    )

    # --- kiểm tra model artifact trước khi cho nhập ---
    status = inference.models_available()
    missing_tasks = [t for t, (ok, _) in status.items() if not ok]
    if missing_tasks:
        for task in missing_tasks:
            _, missing = status[task]
            st.error(
                f"Thiếu model artifact cho **{task}** "
                f"(models/transformer/{task}/). Thiếu: {', '.join(missing)}"
            )
        st.info(
            "Đặt checkpoint đã train vào `models/transformer/sentiment/` và "
            "`models/transformer/topic/` rồi chạy lại. Các tab phân tích khác "
            "vẫn hoạt động bình thường."
        )
        return

    # --- input ---
    raw = st.text_area(
        "Nhập phản hồi",
        height=120,
        placeholder="Ví dụ: Giảng viên dạy rất dễ hiểu và nhiệt tình",
    )

    if not st.button("Dự đoán", type="primary"):
        return

    # --- input validation ---
    if not raw or not raw.strip():
        st.warning("Vui lòng nhập phản hồi trước khi dự đoán.")
        return

    try:
        with st.spinner("Đang tải model và dự đoán (CPU — lần đầu có thể mất vài giây)…"):
            result = inference.predict_feedback(raw)
    except FileNotFoundError as exc:
        st.error("Không load được model artifact.")
        st.code(str(exc))
        return
    except Exception as exc:  # lỗi load/tokenize — không show traceback dài
        st.error("Đã xảy ra lỗi khi dự đoán.")
        st.code(f"{type(exc).__name__}: {exc}")
        return

    sent = result["sentiment"]
    topic = result["topic"]

    # --- output ---
    st.subheader("Kết quả")
    c1, c2 = st.columns(2)
    c1.metric("Sentiment", sent["label"], f"{sent['confidence']*100:.2f}%")
    c2.metric("Topic", topic["label"], f"{topic['confidence']*100:.2f}%")

    if result["truncated"]:
        st.caption(
            "Lưu ý: văn bản dài hơn `max_length=32` đã bị cắt bớt khi tokenize."
        )

    with st.expander("Chi tiết xác suất từng lớp"):
        col_s, col_t = st.columns(2)
        with col_s:
            st.markdown("**Sentiment**")
            st.dataframe(
                [{"Lớp": k, "Xác suất": f"{v*100:.2f}%"} for k, v in sent["probs"].items()],
                use_container_width=True, hide_index=True,
            )
        with col_t:
            st.markdown("**Topic**")
            st.dataframe(
                [{"Lớp": k, "Xác suất": f"{v*100:.2f}%"} for k, v in topic["probs"].items()],
                use_container_width=True, hide_index=True,
            )

    with st.expander("Văn bản sau tiền xử lý (text_clean)"):
        st.code(result["text_clean"])

    st.caption(
        "Kết quả từ PhoBERT (MỨC 1). Xác suất là softmax của model, "
        "chưa qua calibration. Model chạy trên CPU."
    )
