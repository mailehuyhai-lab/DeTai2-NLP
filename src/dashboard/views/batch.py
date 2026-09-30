"""Batch CSV prediction view for the MỨC 1 dashboard."""
from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from .. import inference


_OUTPUT_COLUMNS = [
    "predicted_sentiment",
    "sentiment_confidence",
    "predicted_topic",
    "topic_confidence",
    "text_clean",
    "truncated",
    "prediction_status",
    "prediction_error",
]


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV in memory, trying common Unicode encodings."""
    if uploaded_file is None:
        raise ValueError("Chưa chọn file CSV.")
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("File CSV rỗng.")

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258"):
        try:
            frame = pd.read_csv(BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise ValueError(
            "Không đọc được mã hóa file CSV. Hãy lưu file ở UTF-8."
        ) from last_error

    if len(frame.columns) == 0:
        raise ValueError("CSV không có header/cột dữ liệu.")
    if frame.empty:
        raise ValueError("CSV chỉ có header, chưa có dòng dữ liệu.")
    return frame


def choose_default_text_column(columns) -> str | None:
    """Choose only a semantically named text column; never guess unrelated data."""
    columns = list(columns)
    for preferred in ("text", "text_clean"):
        if preferred in columns:
            return preferred
    string_like = [
        column for column in columns
        if any(token in str(column).lower() for token in ("text", "feedback", "comment", "review"))
    ]
    return string_like[0] if len(string_like) == 1 else None


def _empty_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in _OUTPUT_COLUMNS:
        result[column] = pd.NA
    return result


def build_batch_results(
    frame: pd.DataFrame,
    text_column: str,
    predictor=inference.predict_feedback_batch,
) -> pd.DataFrame:
    """Add prediction columns while preserving all source columns and row order."""
    if text_column not in frame.columns:
        raise ValueError(f"Không tìm thấy cột văn bản: {text_column}")
    if frame.empty:
        raise ValueError("CSV không có dòng dữ liệu để phân tích.")

    result = _empty_output_columns(frame)
    records = predictor(frame[text_column].tolist())
    if len(records) != len(frame):
        raise ValueError("Số kết quả prediction không khớp số dòng CSV.")

    for index, record in enumerate(records):
        result.at[index, "prediction_status"] = record["status"]
        result.at[index, "prediction_error"] = record["error"]
        prediction = record.get("result")
        if prediction is None:
            continue
        sentiment = prediction["sentiment"]
        topic = prediction["topic"]
        result.at[index, "predicted_sentiment"] = sentiment["label"]
        result.at[index, "sentiment_confidence"] = sentiment["confidence"]
        result.at[index, "predicted_topic"] = topic["label"]
        result.at[index, "topic_confidence"] = topic["confidence"]
        result.at[index, "text_clean"] = prediction["text_clean"]
        result.at[index, "truncated"] = prediction["truncated"]
    return result


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize results as UTF-8 with BOM for reliable spreadsheet import."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def render() -> None:
    st.header("Batch Analysis")
    st.caption(
        "Upload CSV → chọn cột văn bản → chạy cùng pipeline PhoBERT của Single Analysis "
        "→ xem và tải kết quả. File chỉ được xử lý trong bộ nhớ dashboard."
    )

    uploaded = st.file_uploader("Chọn file CSV", type=["csv"], key="batch_csv_upload")
    if uploaded is None:
        st.info("Chọn một file CSV để bắt đầu.")
        return

    try:
        frame = read_uploaded_csv(uploaded)
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return

    default_column = choose_default_text_column(frame.columns)
    if default_column is None:
        st.warning(
            "Không tự nhận diện được cột văn bản. Hãy chọn cột chứa phản hồi "
            "bằng danh sách bên dưới."
        )
    text_column = st.selectbox(
        "Cột chứa phản hồi",
        options=list(frame.columns),
        index=(list(frame.columns).index(default_column) if default_column else 0),
        key="batch_text_column",
    )
    st.caption(f"Đã đọc {len(frame):,} dòng và {len(frame.columns):,} cột.")
    st.dataframe(frame.head(10), use_container_width=True, hide_index=True)

    if st.button("Run Analysis", type="primary", key="run_batch_analysis"):
        with st.spinner("Đang chạy batch trên CPU; lần đầu có thể mất vài giây…"):
            try:
                results = build_batch_results(frame, text_column)
            except FileNotFoundError as exc:
                st.error("Thiếu checkpoint PhoBERT local, không thể chạy batch.")
                st.code(str(exc))
                return
            except Exception as exc:
                st.error("Batch prediction thất bại; không tạo kết quả một phần.")
                st.code(f"{type(exc).__name__}: {exc}")
                return
        st.session_state["m1_batch_results"] = results
        st.session_state["m1_batch_text_column"] = text_column

    results = st.session_state.get("m1_batch_results")
    if results is None:
        return

    st.subheader("Kết quả Batch Analysis")
    success_count = int((results["prediction_status"] == "success").sum())
    st.metric("Dòng dự đoán thành công", f"{success_count:,}/{len(results):,}")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=dataframe_csv_bytes(results),
        file_name="batch_prediction_results.csv",
        mime="text/csv",
        key="download_batch_csv",
    )
    st.caption(
        "sentiment_confidence/topic_confidence là xác suất softmax của model, "
        "chưa qua calibration."
    )
    if st.button("Xóa kết quả batch hiện tại", key="clear_batch_results"):
        del st.session_state["m1_batch_results"]
        st.rerun()
