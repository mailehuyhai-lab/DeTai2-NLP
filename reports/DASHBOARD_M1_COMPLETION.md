# Dashboard MỨC 1 — Hoàn thiện các chức năng phân tích

## 1. Scope

Phạm vi thực hiện là **MỨC 1 only** trên dashboard Streamlit của project UIT-VSFC. Các thay đổi chỉ nằm trong `src/dashboard/`, `tests/`, `README.md` và báo cáo này. Không huấn luyện lại model, không thay đổi `data/raw/`, `data/processed/`, `results/`, `models/` và không triển khai Ontology/Mức 2.

## 2. Existing features

Dashboard hiện có các view phân tích được giữ nguyên:

- Tổng quan
- Phân bố nhãn
- Sentiment × Topic
- Từ khóa & Emoji
- Độ dài văn bản
- So sánh mô hình
- Phân tích lỗi
- Demo dự đoán / Single Analysis
- Thời gian (thông báo không khả dụng vì dataset không có timestamp)

## 3. Newly implemented features

Các chức năng MỨC 1 mới được bổ sung:

- Batch Analysis từ file CSV.
- Download CSV cho kết quả batch.
- Model Info đọc trực tiếp từ artifact Transformer.
- Error / Low Confidence cho test errors và batch confidence thật.

## 4. Batch Analysis

- Input qua `st.file_uploader`, chỉ nhận `.csv`.
- `pandas` đọc file trong bộ nhớ; thử `utf-8-sig`, `utf-8`, `cp1258` và báo lỗi rõ khi không đọc được encoding.
- Tự chọn cột text khi có `text`, `text_clean` hoặc một cột duy nhất có tên liên quan text/feedback/comment/review; nếu không nhận diện được thì yêu cầu người dùng chọn.
- Giữ nguyên tất cả cột nguồn và thứ tự dòng.
- Bổ sung các cột:
  - `predicted_sentiment`
  - `sentiment_confidence`
  - `predicted_topic`
  - `topic_confidence`
  - `text_clean`
  - `truncated`
  - `prediction_status`
  - `prediction_error`
- Mỗi dòng đi qua cùng `clean_text()` và checkpoint PhoBERT của Single Analysis.
- Dòng input rỗng/NaN được đánh dấu `invalid_input`; lỗi model/tokenizer không được nuốt để tránh biến batch một phần thành kết quả đáng tin.

## 5. Export

- Nút `Download CSV` xuất `batch_prediction_results.csv`.
- Dữ liệu được serialize UTF-8-SIG trong bộ nhớ để tiếng Việt/emoji mở đúng bằng pandas và Excel.
- Không sửa file upload gốc; không ghi output xuống `data/` hoặc `results/`.
- Không thêm dependency mới; Excel không phải dependency bắt buộc.

## 6. Model Info

View đọc `results/transformer/sentiment_metrics.json` và `results/transformer/topic_metrics.json` qua loader. Các giá trị hiển thị là giá trị artifact thật:

| Task | Model | Accuracy test | Macro-F1 test | Weighted-F1 test |
|---|---|---:|---:|---:|
| Sentiment | `vinai/phobert-base-v2` | 0.9198 | 0.8224 | 0.9226 |
| Topic | `vinai/phobert-base-v2` | 0.8506 | 0.7673 | 0.8603 |

Các thông tin bổ sung từ artifact: classes, `seed=42`, `max_length=32`, `batch_size=64`, `learning_rate=2e-5`, `epochs_trained=2`, `best_epoch=2`, `best_dev_macro_f1`, dev metrics và class weights. Trường thiếu được báo là không có trong artifact thay vì nhận giá trị bịa.

## 7. Error / Low Confidence

- Test artifact Transformer có schema thật:
  `id`, `text`, `true_label`, `predicted_label`, `correct`, `true_label_name`, `predicted_label_name`.
- Artifact test không chứa logits, xác suất hoặc confidence. Dashboard không bịa confidence và không suy diễn confidence từ metric/class weights.
- Bộ lọc test chỉ dùng `correct` để hiển thị errors; có thể lọc thêm theo true/predicted label và download CSV.
- Low-confidence chỉ áp dụng cho kết quả Batch Analysis trong `st.session_state` vì các kết quả này có confidence softmax trực tiếp từ model.
- Ngưỡng confidence được chọn bằng slider, đã kiểm thử 0.50, 0.60 và 0.80.

## 8. Test cases

Đã chạy các kiểm tra:

- Compile dashboard và tests.
- Import toàn bộ dashboard modules.
- Đọc CSV Unicode/emoji.
- Từ chối file bytes rỗng.
- Từ chối CSV chỉ có header.
- Yêu cầu chọn cột khi không nhận diện được cột text.
- Giữ cột gốc và đánh dấu các dòng blank/NaN.
- Batch delegation sang `predict_feedback()`.
- Export UTF-8-SIG và reload bằng pandas.
- Test artifact không claim confidence.
- Filter test errors chỉ giữ `correct == 0`.
- Filter batch low-confidence chỉ dùng cột confidence thật.
- Smoke test Streamlit app qua browser preview và kiểm tra navigation `Batch Analysis`, `Model Info`, `Error / Low Confidence`.

## 9. Test results

- `python -m unittest discover -s tests -v`: **10/10 pass**.
- `python -m compileall -q src/dashboard tests`: pass.
- Single Analysis thực tế trên `Giảng viên dạy rất dễ hiểu và nhiệt tình`: sentiment `Positive` (≈0.954), topic `Lecturer` (≈0.921), không truncated.
- Batch 4 dòng thực tế: 4/4 `success`, đủ bốn cột prediction/confidence, CSV UTF-8-SIG reload được.
- Model Info đọc đúng test metrics từ artifact.
- Test errors lọc đúng số dòng `correct == 0` (sentiment: 254; topic: 473).
- Streamlit startup không exception; các view cũ và mới render trong browser preview. Sau khi chuẩn hóa cột metadata hiển thị thành chuỗi, Model Info render không còn lỗi Arrow serialization trong log smoke test.

## 10. Known limitations

- Confidence là softmax trực tiếp, chưa calibration.
- Test prediction artifacts không có confidence nên không thể lọc low-confidence test một cách trung thực.
- Batch upload thường không có true label; `Errors only` và `Low confidence + errors` chỉ có ý nghĩa khi CSV có cột nhãn thật tương thích.
- `max_length=32` làm văn bản dài bị truncation.
- Dataset không có timestamp nên view Thời gian chỉ thông báo không khả dụng.
- Checkpoint local trong `models/transformer/` không được commit; dashboard cần người dùng chuẩn bị checkpoint.
- Không triển khai Ontology/Mức 2: không OWL/TTL, Protégé, RDF, SPARQL, SWRL, ontology mapper, rule engine, hybrid fusion, ablation Mức 2, GraphRAG, LLM explanation, database hoặc REST API riêng.
