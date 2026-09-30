# M1 COMPLETENESS AUDIT — Đề tài #2 (UIT-VSFC)

**Đề tài:** Phân tích cảm xúc và chủ đề từ phản hồi người dùng bằng NLP kết hợp Ontology cảm xúc để hỗ trợ cải tiến dịch vụ
**Nhóm:** NLP — Phản hồi người dùng
**Phạm vi audit:** Mức 1 (Thực tập ngành) — đối chiếu với `reference/De_cuong_de_tai_CuChi_2026.docx`, mục "2. Mức 1 — Kết quả Thực tập ngành", "4. Hướng dẫn thực hiện và tiêu chí kỹ thuật", "6. Yêu cầu báo cáo và bàn giao" và "Hướng dẫn chung" của đề cương.
**Ngày audit:** 2026-09-30
**Tính chất:** Read-only audit. Không sửa code, dataset, model, results hay reports hiện có.

---

## 1. Executive Summary

Mức 1 của đề tài #2 **đáp ứng đầy đủ phần lõi bắt buộc** theo đề cương:

- Corpus phản hồi tiếng Việt (UIT-VSFC) đã được thu thập, chuẩn hóa nhãn sentiment/topic, kiểm tra toàn vẹn MD5.
- Pipeline đầy đủ: Preprocessing → EDA → Baseline TF-IDF + Logistic Regression/Linear SVM → PhoBERT Transformer → Error Analysis → Dashboard Streamlit.
- Demo web (Streamlit) nhận phản hồi và trả sentiment + topic + confidence — đáp ứng yêu cầu "API/web demo" ở nhánh **web demo** (Streamlit nằm trong tài liệu tham khảo chính thức của đề cương — mục [8]).
- Báo cáo lỗi điển hình, confusion matrix, Macro-F1/Weighted-F1 cho **cả hai task riêng biệt** đều có artifact.

**Tổng kết:** 13 yêu cầu Mức 1 được đối chiếu → **11 PASS, 2 PARTIAL, 0 MISSING.**

Hai điểm PARTIAL đều là giới hạn có thể giải trình hoặc xử lý tối thiểu bằng tài liệu, **không phải lỗi kỹ thuật**:

1. **Dashboard "theo thời gian"** — UIT-VSFC không có trường timestamp; đã xử lý trung thực bằng tab "Thời gian" ghi rõ limitation, không tạo dữ liệu giả. Đây là giới hạn dữ liệu, không thể đáp ứng trực tiếp.
2. **"Dữ liệu mẫu hoặc script tải dữ liệu công khai"** — `data/processed/` đã được commit (đáp ứng phần "dữ liệu mẫu"), nhưng README chỉ nói chung chung "nguồn công khai UIT-VSFC" mà **chưa ghi URL cụ thể** (`https://github.com/kietnv/uit-vsfc` — có trong đề cương mục 5[1]) và chưa có script tải. Xử lý tối thiểu: bổ sung URL vào README (chỉ sửa tài liệu).

Mức 1 **sẵn sàng demo và sẵn sàng viết báo cáo**, với 2 ghi chú cần nêu rõ khi bảo vệ.

---

## 2. Bảng đối chiếu toàn bộ yêu cầu Mức 1

| # | Yêu cầu (nguồn trong đề cương) | Trạng thái | Bằng chứng chính |
|---|---|---|---|
| R1 | Thu thập/sử dụng corpus phản hồi tiếng Việt; chuẩn hóa nhãn sentiment và topic (§2, bullet 1) | **PASS** | `data/raw/` (9 file UIT-VSFC), `data/processed/{train,dev,test}.csv` (16.173 mẫu), `label_maps.json` |
| R2 | Tiền xử lý (§2, bullet 2) | **PASS** | `src/preprocessing/`, `reports/PREPROCESSING_PLAN.md`, `PREPROCESSING_RESULT.md`, `preprocessing_log.json` |
| R3 | EDA (§2, bullet 2) | **PASS** | `src/eda/`, `results/eda/` (8 figures, 8 tables), `reports/EDA_RESULT.md` |
| R4 | Baseline TF-IDF + Logistic Regression/SVM (§2, bullet 2) | **PASS** | `src/baseline/`, `models/baseline/` (5 joblib), `results/baseline/model_comparison.csv` |
| R5 | Thử mô hình Transformer phù hợp (PhoBERT/BERT/DistilBERT) (§2, bullet 3) | **PASS** | `src/transformer/`, `models/transformer/{sentiment,topic}/` (PhoBERT-base-v2), `results/transformer/*_metrics.json` |
| R6 | Dashboard theo thời gian, chủ đề, mức sentiment, từ khóa điển hình (§2, bullet 4) | **PARTIAL** | 8 tab phân tích + tab demo (`src/dashboard/`). "Theo thời gian" không khả thi — dataset không có timestamp; tab `time_notice.py` ghi limitation trung thực |
| R7 | API/web demo nhận phản hồi, trả sentiment/topic (§2, bullet 5) | **PASS** | Tab "Demo dự đoán": `src/dashboard/views/prediction.py` + `src/dashboard/inference.py` |
| R8 | Chuẩn hóa Unicode, xử lý emoji/teencode, tách từ tiếng Việt (§4, bullet 1) | **PARTIAL** | Unicode NFC ✓, emoji acronym → Unicode ✓ (`text_cleaner.py`). Teencode không có từ điển riêng; "tách từ" được thừa hưởng từ dữ liệu pre-tokenized của UIT-VSFC + BPE của PhoBERT |
| R9 | Đánh giá sentiment và topic **riêng**, không gộp một chỉ số (§4, bullet 3) | **PASS** | Model, metrics, confusion matrix, predictions tách biệt cho từng task |
| R10 | Lưu ví dụ false positive/false negative để phân tích định tính (§4, bullet 4) | **PASS** | `results/error_analysis/{sentiment,topic}_error_samples.csv` (30 + 40 mẫu) |
| R11 | Metric: Macro-F1/Weighted-F1 + confusion matrix (§4, "Đánh giá") | **PASS** | `model_comparison.csv` (baseline + transformer), `*_confusion_matrix.csv` |
| R12 | Baseline cho mọi mô hình; train/val/test; chống leakage; lưu seed + cấu hình (Hướng dẫn chung §4) | **PASS** | Split gốc UIT-VSFC giữ nguyên; `seed=42`, `max_length`, `lr`, `batch_size`, class weights lưu trong `*_metrics.json` |
| R13 | Bàn giao: README, requirements/environment, hướng dẫn chạy, dữ liệu mẫu hoặc script tải dữ liệu công khai; báo cáo mô tả dữ liệu/kiến trúc/thuật toán/thí nghiệm/kết quả/lỗi (§6) | **PARTIAL** | README 16 mục ✓, `requirements.txt` ✓, hướng dẫn chạy đầy đủ ✓, `data/processed/` committed ✓; **thiếu URL/script tải raw data** |

**Đếm:** PASS = 11 · PARTIAL = 2 (R6, R8, R13 → xem mục 4; R13 gộp nhiều tiểu mục — chỉ tiểu mục "script tải dữ liệu" partial) · MISSING = 0.

*Lưu ý đếm:* R13 là yêu cầu tổng hợp; chỉ phần "script tải dữ liệu công khai hoặc đường dẫn cụ thể" là PARTIAL, các phần còn lại PASS.

---

## 3. Chi tiết từng yêu cầu

### R1 — Corpus phản hồi + chuẩn hóa nhãn — **PASS**

- **Yêu cầu:** "Thu thập hoặc sử dụng corpus phản hồi tiếng Việt/tiếng Anh; chuẩn hóa nhãn sentiment và topic."
- **Bằng chứng:**
  - `data/raw/{train,dev,test}/{sents,sentiments,topics}.txt` — đúng định dạng UIT-VSFC gốc (kèm `data/raw/README.txt` mô tả nhãn).
  - `data/processed/stats_summary.json`: train 11.424 / dev 1.583 / test 3.166 = **16.173 mẫu**; phân bố sentiment {0: Negative, 1: Neutral, 2: Positive} và topic {0: Lecturer, 1: Training_program, 2: Facility, 3: Others}.
  - `data/processed/label_maps.json` — chuẩn hóa tên nhãn thống nhất toàn pipeline.
  - `preprocessing_log.json`: MD5 checksum 9 file raw khớp; loại 2 dòng annotation conflict (train_11293, train_11417 — cùng text, nhãn sentiment mâu thuẫn).
- **Cách chứng minh khi demo:** mở `reports/DATASET_AUDIT.md` + tab "Tổng quan"/"Phân bố nhãn" của dashboard.

### R2 — Tiền xử lý — **PASS**

- **Yêu cầu:** "Tiền xử lý" (trong bullet "Tiền xử lý, EDA và xây baseline…").
- **Bằng chứng:** `src/preprocessing/text_cleaner.py` (`clean_text()`: NFC → emoji acronym → whitespace; `to_lowercase()` cho baseline), `build_processed.py`, `validate.py`; kết quả `data/processed/` + `reports/PREPROCESSING_RESULT.md`.
- **Cách chứng minh:** chạy `python -m src.preprocessing.validate`; expander "Văn bản sau tiền xử lý (text_clean)" trong tab Demo dự đoán cho thấy preprocessing trực tiếp.

### R3 — EDA — **PASS**

- **Bằng chứng:** `src/eda/{analyze,plots,run_eda,validate}.py`; `results/eda/figures/` (8 PNG: phân bố sentiment/topic, heatmap sentiment×topic, độ dài, top keywords, emoji frequency); `results/eda/tables/` (8 CSV); `reports/EDA_RESULT.md`.
- **Cách chứng minh:** các tab "Phân bố nhãn", "Sentiment × Topic", "Từ khóa & Emoji", "Độ dài văn bản" render trực tiếp artifact này.

### R4 — Baseline TF-IDF + Logistic Regression/SVM — **PASS**

- **Bằng chứng:** `src/baseline/{tfidf,train_baseline,evaluate,error_analysis}.py`; `models/baseline/` (tfidf_vectorizer + 4 classifier joblib); `results/baseline/model_comparison.csv`:

  | Task | Model | Accuracy | Macro F1 | Weighted F1 |
  |---|---|---:|---:|---:|
  | Sentiment | Logistic Regression | 0.8664 | 0.7174 | 0.8717 |
  | Sentiment | Linear SVM | 0.8920 | 0.7304 | 0.8872 |
  | Topic | Logistic Regression | 0.8042 | 0.7184 | 0.8213 |
  | Topic | Linear SVM | 0.8588 | 0.7531 | 0.8602 |

- **Cách chứng minh:** `reports/BASELINE_RESULT.md`; tab "So sánh mô hình".

### R5 — Transformer (PhoBERT) — **PASS**

- **Yêu cầu:** "Thử một mô hình Transformer phù hợp (PhoBERT/BERT/DistilBERT) hoặc dùng embedding + classifier."
- **Bằng chứng:** `vinai/phobert-base-v2` fine-tune riêng cho 2 task; `models/transformer/{sentiment,topic}/` (safetensors + tokenizer + `label_map.json`); `results/transformer/{sentiment,topic}_metrics.json`:
  - Sentiment: test acc **0.9198**, macro F1 **0.8224**, weighted F1 **0.9226** (seed 42, max_length 32, lr 2e-5, 2 epochs, class weights).
  - Topic: test acc **0.8506**, macro F1 **0.7673**, weighted F1 **0.8603**.
  - `model_comparison.csv` (transformer) so sánh trực tiếp SVM vs PhoBERT từng metric.
- **Cách chứng minh:** `reports/TRANSFORMER_RESULT.md`; tab "So sánh mô hình"; demo dự đoán dùng chính checkpoint này.

### R6 — Dashboard (thời gian / chủ đề / sentiment / từ khóa) — **PARTIAL**

- **Yêu cầu:** "Xây dashboard theo thời gian, chủ đề, mức sentiment và các từ khóa điển hình."
- **Đã có:** `src/dashboard/` — 9 tab: Tổng quan, Phân bố nhãn, Sentiment × Topic, Từ khóa & Emoji, Độ dài văn bản, So sánh mô hình, Phân tích lỗi, Demo dự đoán, Thời gian.
  - "theo chủ đề" ✓ (Phân bố nhãn, Sentiment × Topic)
  - "mức sentiment" ✓ (Phân bố nhãn, Sentiment × Topic, So sánh mô hình)
  - "từ khóa điển hình" ✓ (Từ khóa & Emoji)
  - "theo thời gian" ✗ — **UIT-VSFC không có trường timestamp.**
- **Cách xử lý hiện tại (đúng nguyên tắc):** `src/dashboard/views/time_notice.py` hiển thị cảnh báo "Phân tích theo thời gian không khả dụng", nói rõ không tạo ngày giả/không suy diễn từ ID. README §13 ghi limitation.
- **Thiếu chính xác:** một chế độ xem theo thời gian — không thể có vì dữ liệu gốc không chứa thời gian.
- **Mức độ ảnh hưởng:** thấp–trung bình. Đây là giới hạn dữ liệu công khai, không phải thiếu sót triển khai; cần **chủ động giải trình** khi báo cáo.
- **Đề xuất xử lý tối thiểu:** giữ nguyên tab limitation; trong báo cáo nêu rõ "yêu cầu theo thời gian không khả thi với UIT-VSFC (không có timestamp); nếu Mức 2 dùng phản hồi nội bộ có timestamp thì bổ sung được". Không tự tạo timestamp.

### R7 — API/web demo — **PASS (ở nhánh "web demo")**

- **Yêu cầu:** "Viết API/web demo nhận phản hồi và trả sentiment/topic." — đề cương dùng dấu "/", nghĩa là **một trong hai**; Streamlit là tài liệu tham khảo chính thức của đề tài (§5[8]).
- **Đáp ứng:** tab "Demo dự đoán" — input → `clean_text()` → PhoBERT sentiment + topic → nhãn + softmax confidence (kèm disclaimer "chưa calibration"); kiểm tra artifact trước khi load; xử lý input rỗng; lazy-load qua `st.cache_resource`.
- **Đã kiểm chứng:** known sample "Giảng viên dạy rất dễ hiểu và nhiệt tình" → Positive 95.38% / Lecturer 92.07%; 20/20 mẫu test khớp với `results/transformer/*_test_predictions.csv` cho cả hai task.
- **Diễn giải không nâng yêu cầu:** chưa có REST API riêng — **không bắt buộc** ở Mức 1 vì nhánh "web demo" đã được đáp ứng. Nếu GVHD yêu cầu rõ API thì đó là việc bổ sung nhỏ (FastAPI bọc `predict_feedback`), không thuộc nghĩa vụ M1 hiện tại.

### R8 — Unicode / emoji / teencode / tách từ — **PARTIAL**

- **Yêu cầu (§4):** "Chuẩn hóa Unicode, xử lý emoji/teencode và tách từ tiếng Việt."
- **Đã có:** NFC Unicode normalization ✓; 19 emoji acronym → Unicode ✓ (`EMOJI_ACRONYM_MAP`, dài-trước-ngắn-sau); whitespace ✓.
- **Chưa có:** từ điển teencode riêng (ví dụ "qá", "vcl", "k" → "không" không được chuẩn hóa); không dùng thư viện tách từ (pyvi/underthesea/VnTokenizer).
- **Giải trình hợp lệ:** UIT-VSFC **đã pre-tokenized** sẵn (dấu câu tách bằng khoảng trắng) — pipeline giữ nguyên theo quyết định Q3=A đã duyệt; PhoBERT dùng BPE riêng, không cần word segmentation ngoài. Teencode: VSFC chủ yếu chứa acronym emoji đã map; teencode dạng khác tồn tại trong test input tự do nhưng không nằm trong tập acronym của corpus.
- **Mức ảnh hưởng:** thấp — chủ yếu ảnh hưởng demo với input tự do nhiều teencode.
- **Xử lý tối thiểu (tùy chọn, doc-level):** ghi trong báo cáo/README rằng word segmentation được thừa hưởng từ bản pre-tokenized của UIT-VSFC; teencode ngoài acronym emoji được để nguyên để không phá PhoBERT subword. Không cần code mới cho M1.

### R9 — Đánh giá sentiment/topic riêng — **PASS**

- **Bằng chứng:** 2 pipeline độc lập: `train_task.py --task {sentiment,topic}`; metrics/predictions/confusion matrix tách file (`sentiment_metrics.json` vs `topic_metrics.json`, `sentiment_*_test_predictions.csv` vs `topic_*_test_predictions.csv`); dashboard demo trả 2 nhãn riêng.
- Đúng yêu cầu "không gộp hai nhiệm vụ thành một chỉ số duy nhất".

### R10 — Lưu ví dụ FP/FN — **PASS**

- **Bằng chứng:** `results/error_analysis/sentiment_error_samples.csv` (30 mẫu), `topic_error_samples.csv` (40 mẫu) với cột `text, true_label, predicted_label`; `*_error_summary.csv` xếp hạng cặp nhầm lẫn (Positive→Negative 33.33% lỗi sentiment; Lecturer→Training_program 36.47% lỗi topic); `reports/ERROR_ANALYSIS_RESULT.md`; tab "Phân tích lỗi".

### R11 — Macro-F1/Weighted-F1 + confusion matrix — **PASS**

- **Bằng chứng:** `results/baseline/model_comparison.csv` (macro + weighted P/R/F1), `results/transformer/model_comparison.csv` (SVM vs PhoBERT, kèm F1 từng lớp), `*_confusion_matrix.csv` cho cả dev và test. Topic coherence không áp dụng (không dùng topic modeling — đề cương chỉ yêu cầu "nếu dùng").

### R12 — Baseline, split, chống leakage, seed/config — **PASS**

- **Bằng chứng:** giữ nguyên split gốc UIT-VSFC (không trộn); baseline LR/SVM làm chuẩn so sánh cho PhoBERT; `seed: 42` + `max_length`, `batch_size`, `learning_rate`, `epochs`, `class_weights` lưu trong `results/transformer/*_metrics.json`; `data/processed/class_weights.json`; `preprocessing_log.json` ghi toàn bộ quyết định Q1–Q4.

### R13 — Bàn giao: README / requirements / hướng dẫn chạy / dữ liệu mẫu hoặc script tải / báo cáo — **PARTIAL**

- **Đã có:** `README.md` 16 mục (dataset, pipeline, cấu trúc repo, environment, install, dataset preparation, how to run từng bước + thứ tự pipeline, cấu hình model, kết quả, error analysis, limitations, future work, reproducibility); `requirements.txt` 9 dependencies; `data/processed/` **được commit** (đáp ứng "dữ liệu mẫu" — reviewer clone về chạy được EDA/baseline/dashboard ngay); 8 reports mô tả dữ liệu → tiền xử lý → EDA → baseline → transformer → lỗi → dashboard plan.
- **Thiếu chính xác:** (a) README không ghi **URL cụ thể** của UIT-VSFC (đề cương có: `https://github.com/kietnv/uit-vsfc`); (b) không có script tải raw data → người dùng không tự tái tạo được `data/raw/` + không verify được MD5 từ đầu.
- **Mức ảnh hưởng:** thấp–trung bình. `data/processed/` committed nên toàn bộ pipeline sau preprocessing tái lập được; chỉ bước raw→processed cần người dùng tự tìm dataset.
- **Xử lý tối thiểu:** thêm 1–2 dòng URL vào README §3/§8 (chỉ sửa tài liệu, ~5 phút). Script tải là tùy chọn, không bắt buộc vì đề cương chấp nhận "dữ liệu mẫu **hoặc** script tải" và `data/processed/` đã là dữ liệu mẫu đầy đủ.

---

## 4. Evidence Matrix

| Hạng mục | File / Path | Module / Function | Artifact / Report |
|---|---|---|---|
| Raw corpus UIT-VSFC | `data/raw/{train,dev,test}/*.txt` (9 file) + `data/raw/README.txt` | — | `reports/DATASET_AUDIT.md`, `preprocessing_log.json` (MD5 pass) |
| Processed corpus | `data/processed/{train,dev,test}.csv` — 11.424/1.583/3.166 | `src/preprocessing/build_processed.py` | `stats_summary.json`, `label_maps.json` |
| Preprocessing | `src/preprocessing/text_cleaner.py` | `clean_text()`, `to_lowercase()`, `EMOJI_ACRONYM_MAP` | `reports/PREPROCESSING_PLAN.md` + `PREPROCESSING_RESULT.md` |
| EDA | `src/eda/{analyze,plots,run_eda,validate}.py` | — | `results/eda/figures/` (8 PNG), `results/eda/tables/` (8 CSV), `reports/EDA_RESULT.md` |
| Baseline TF-IDF + LR/SVM | `src/baseline/{tfidf,train_baseline,evaluate}.py` | TfidfVectorizer + LogisticRegression + LinearSVC | `models/baseline/*.joblib`, `results/baseline/model_comparison.csv`, `reports/BASELINE_RESULT.md` |
| Transformer | `src/transformer/{config,dataset,model,train_task,train_all,evaluate,validate,metrics,tokenize_stats}.py` | PhoBERT `vinai/phobert-base-v2`, seed 42, max_length 32 | `models/transformer/{sentiment,topic}/`, `results/transformer/*_metrics.json`, `reports/TRANSFORMER_RESULT.md` |
| Error analysis | `src/baseline/error_analysis.py` | — | `results/error_analysis/` (5 file), `reports/ERROR_ANALYSIS_RESULT.md` |
| Dashboard | `src/dashboard/{app,loader,inference}.py` + `views/` (10 file) | `render()` per tab; `predict_feedback()` | `reports/DASHBOARD_PHASE2_PLAN.md` |
| Web demo (R7) | `src/dashboard/views/prediction.py` + `inference.py` | `models_available()`, `get_model_bundle()`, `predict_feedback()` | verified: known sample Positive 95.38%/Lecturer 92.07%; 20/20 agreement |
| Hướng dẫn chạy | `README.md` §7–§9 | — | `requirements.txt` (9 deps) |
| Tái lập | seed 42, MD5 raw, processed committed | — | `*_metrics.json`, `preprocessing_log.json` |

---

## 5. Missing / Partial Items

| Mục | Trạng thái | Thiếu chính xác | Ảnh hưởng | Xử lý tối thiểu |
|---|---|---|---|---|
| R6 — Dashboard theo thời gian | PARTIAL (giới hạn dữ liệu) | Không có time-series view; UIT-VSFC không có timestamp | Thấp–TB; phải giải trình khi bảo vệ | Giữ tab limitation; nêu trong báo cáo. **Tuyệt đối không tạo timestamp giả** |
| R8 — Teencode + tách từ | PARTIAL | Không có từ điển teencode; không dùng thư viện word-segmentation | Thấp (data đã pre-tokenized; PhoBERT BPE) | Ghi giải thích trong báo cáo: pre-tokenized + BPE thay cho tách từ ngoài |
| R13 — Script tải/URL dữ liệu | PARTIAL | README chưa có URL `github.com/kietnv/uit-vsfc`; không có script tải raw | Thấp–TB (processed data đã commit) | Thêm URL vào README §3/§8 (doc-only, ~5 phút). Script tùy chọn |

**Không có mục MISSING.** Không phát hiện blocker kỹ thuật nào cho việc đóng Mức 1.

---

## 6. Demo Readiness

| Kịch bản demo | Sẵn sàng | Cách trình diễn |
|---|---|---|
| Chạy dashboard | ✅ | `streamlit run src/dashboard/app.py` — 9 tab render không lỗi (đã kiểm AppTest) |
| Demo dự đoán trực tiếp | ✅ | Tab "Demo dự đoán": nhập feedback → Sentiment + Topic + confidence; known sample cho kết quả ổn định (Positive 95.38%/Lecturer) |
| Trình bày phân bố & insight | ✅ | Tabs Tổng quan/Phân bố nhãn/Crosstab/Từ khóa/Độ dài đọc trực tiếp `results/eda/` |
| So sánh baseline vs Transformer | ✅ | Tab "So sánh mô hình" + `model_comparison.csv` |
| Trình bày phân tích lỗi | ✅ | Tab "Phân tích lỗi" + `error_analysis/` artifacts |
| Câu hỏi "vì sao không có theo thời gian?" | ✅ (có sẵn câu trả lời) | Tab "Thời gian" ghi limitation; trả lời: dataset không có timestamp, không fabricate |
| Edge cases input demo | ✅ | Input rỗng → warning; emoji/teencode nhẹ → xử lý; input dài >32 tokens → flag truncated |

**Lưu ý vận hành:** lần predict đầu mỗi task ~3s trên CPU (load checkpoint ~517MB/model); checkpoint nằm ngoài Git — máy demo phải có sẵn `models/transformer/`.

## 7. Report Readiness

| Nội dung báo cáo đề cương yêu cầu (§6 + gợi ý cấu trúc) | Nguồn sẵn có |
|---|---|
| Mô tả dữ liệu | `DATASET_AUDIT.md`, `stats_summary.json`, README §3 |
| Kiến trúc hệ thống | README §4–§5, `DASHBOARD_PHASE2_PLAN.md` |
| Thuật toán | `BASELINE_RESULT.md`, `TRANSFORMER_RESULT.md` (config đầy đủ: seed/lr/max_length/epochs/class weights) |
| Thí nghiệm & kết quả | `model_comparison.csv` (2 file), metrics JSON, confusion matrices |
| Phân tích lỗi | `ERROR_ANALYSIS_RESULT.md`, error samples/summaries |
| Bảng so sánh mô hình | `results/transformer/model_comparison.csv` (SVM vs PhoBERT, per-class F1) |
| Hạn chế & hướng phát triển | README §13–§14 + audit này (R6/R8/R13) |

Báo cáo Mức 1 có thể viết ngay từ artifact hiện có. Phần "ít nhất một bảng so sánh mô hình" theo hướng dẫn chung đã có sẵn dữ liệu (baseline vs PhoBERT, cả hai task).

## 8. Final checklist trước khi đóng Mức 1

- [x] Corpus phản hồi tiếng Việt (UIT-VSFC) thu thập + chuẩn hóa nhãn sentiment/topic
- [x] Preprocessing (Unicode NFC, emoji acronym, whitespace; giữ pre-tokenized)
- [x] EDA đầy đủ (figures + tables + report)
- [x] Baseline TF-IDF + Logistic Regression + Linear SVM, đánh giá đầy đủ
- [x] Transformer (PhoBERT-base-v2) fine-tune cho cả 2 task
- [x] Đánh giá sentiment/topic **riêng biệt**, Macro/Weighted-F1 + confusion matrix
- [x] Error analysis với ví dụ FP/FN lưu file
- [x] Dashboard: chủ đề, sentiment, từ khóa, so sánh, lỗi, demo dự đoán
- [x] Web demo nhận feedback → trả sentiment + topic (Streamlit — nhánh "web demo" của yêu cầu)
- [x] README + requirements + hướng dẫn chạy từng bước
- [x] Dữ liệu mẫu (processed) commit; raw/model gitignore; seed + config lưu lại
- [x] Limitation "không có timestamp" được ghi trung thực (dashboard tab + README §13)
- [ ] **[Tùy chọn — nên làm]** Bổ sung URL `https://github.com/kietnv/uit-vsfc` vào README §3/§8 (khắc phục R13 hoàn toàn, chỉ sửa tài liệu)
- [ ] **[Khi bảo vệ]** Chuẩn bị câu trả lời cho: dashboard theo thời gian (giới hạn dữ liệu), tách từ/teencode (pre-tokenized + BPE), "API hay web demo" (đã chọn web demo theo đề cương)
- [ ] **[Mức 2 — chưa làm]** Ontology OWL, rule fusion, giải thích, ablation — đúng phạm vi, chưa triển khai ở M1

---

*Audit read-only: không file nào khác bị thay đổi; không commit, không push.*
