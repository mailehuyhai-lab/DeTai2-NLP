# DASHBOARD PHASE 2 — PLAN (Audit + Implementation Plan)

Phạm vi: **MỨC 1**. Chỉ thêm chức năng **demo inference** trên Streamlit dashboard
(nhập feedback → preprocess → PhoBERT sentiment + topic → hiển thị label + confidence).
Không Ontology / KG / RAG / rule fusion / API server / database.

Audit date: 2026-09-30. Git base: `c9394f9` (sạch, đã push origin/main).

---

## 1. Audit Summary

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Sentiment model | ✅ `models/transformer/sentiment/` | Full checkpoint: `model.safetensors` (540 MB), `config.json`, `label_map.json`, tokenizer files. Git-ignored (`models/`, `*.safetensors`). |
| Topic model | ✅ `models/transformer/topic/` | Cùng cấu trúc. |
| Tokenizer | ✅ lưu trong mỗi checkpoint | `AutoTokenizer.from_pretrained(model_dir)` load được (đã verify; `validate.py` dùng đúng pattern này). |
| Label maps | ✅ 3 nguồn nhất quán | `models/transformer/{task}/label_map.json`, `data/processed/label_maps.json`, `config.SENTIMENT_MAP/TOPIC_MAP`. **Lưu ý:** `config.json` trong checkpoint chỉ có `id2label: LABEL_0/1/2` generic → phải decode bằng `label_map.json`, không dựa vào `config.id2label`. |
| Preprocessing | ✅ tái sử dụng trực tiếp | `src/preprocessing/text_cleaner.clean_text()` — đúng hàm đã sinh `text_clean`. Pure function, read-only. |
| Inference code | ⚠️ batch-only | `evaluate.py` + `WeightedLossTrainer.predict()` chỉ nhận dataloader. Chưa có hàm single-text → cần wrapper mỏng, KHÔNG sửa code cũ. |
| Dashboard | ✅ Phase 1 | `app.py` radio-based routing; thêm 1 entry + 1 nhánh `elif`. `loader.py` không cần sửa. |
| Class weights | ✅ không cần | Chỉ dùng trong loss khi train; inference bỏ qua. |
| CPU inference | ✅ khả thi | Đo thực tế: cold load + predict sentiment ≈ **2.9 s**; forward sau warmup < 1 s/mẫu. |
| Confidence | ✅ an toàn | Softmax trên logits — đọc output có sẵn, không đổi contract. Phải ghi nhãn là "xác suất softmax của model", không claim calibration. |

## 2. Current Repository Evidence

Đã kiểm tra trực tiếp (không đoán):

```
models/transformer/sentiment/
    model.safetensors          540 MB
    config.json                (id2label generic LABEL_* — không dùng để decode)
    label_map.json             {"0":"Negative","1":"Neutral","2":"Positive"}
    tokenizer_config.json, vocab.txt, bpe.codes, added_tokens.json
models/transformer/topic/
    (cùng cấu trúc)
    label_map.json             {"0":"Lecturer","1":"Training_program","2":"Facility","3":"Others"}

src/preprocessing/text_cleaner.py
    clean_text()               NFC → emoji acronym→Unicode → whitespace (giữ case)
    to_lowercase()             (chỉ cho baseline — Phase 2 KHÔNG dùng)

src/transformer/config.py
    MAX_LENGTH = 32, MODELS_DIR, SENTIMENT_MAP, TOPIC_MAP, MODEL_NAME
src/transformer/dataset.py
    FeedbackDataset.__getitem__ — tokenize: max_length, padding=max_length,
    truncation=True, return_tensors="pt"  ← pattern inference phải bắt chước
src/transformer/evaluate.py    AutoTokenizer/AutoModel.from_pretrained(model_dir)
src/transformer/validate.py    xác nhận cả 2 checkpoint load được
src/transformer/model.py       predict() trả logits → softmax cho confidence

src/dashboard/app.py           radio list + elif routing (điểm chèn duy nhất)
src/dashboard/loader.py        BASE_DIR = parents[2], _need(), cache pattern
src/dashboard/views/*.py       8 views Phase 1 (không đụng)

.gitignore                     dòng 78: models/ ; *.safetensors; data/raw/; reference/
requirements.txt               torch>=2.4, transformers>=5.0, streamlit>=1.36 (đủ hết)
```

**Live smoke test đã chạy trong audit** (read-only, không ghi file):
input `"Giảng viên dạy rất dễ hiểu và nhiệt tình"` → `clean_text` → PhoBERT
→ `Positive`, softmax `{Negative:0.022, Neutral:0.024, Positive:0.954}` — đúng kỳ vọng.

**Tokenizer consistency check:** `add_prefix_space=True` (dùng trong EDA) vs mặc định
(dùng trong training) cho ra **id giống hệt** với PhoBERT BPE → không có lệch
preprocessing giữa train/inference. Phase 2 dùng mặc định (= training).

## 3. Inference Architecture

```
user text (raw, giữ case)
  → validate input (non-empty sau strip)
  → clean_text(raw)                         [tái sử dụng src/preprocessing]
  → AutoTokenizer.from_pretrained(model_dir)(text_clean, max_length=32,
        padding="max_length", truncation=True, return_tensors="pt")
  → sentiment model (models/transformer/sentiment) → logits → argmax → label_map.json
                                                        → softmax → confidence
  → topic model     (models/transformer/topic)     → tương tự
  → render: label + confidence (%) trên Streamlit
```

- `device = torch.device("cpu")`, `model.eval()`, `torch.no_grad()`.
- Decode label từ `label_map.json` trong checkpoint (không dùng `config.id2label`).
- Không dùng `text_lower`, không stemming/stopword — đúng preprocessing contract.

## 4. Files To Create

| File | Vai trò |
|---|---|
| `src/dashboard/inference.py` | Cache + inference: `get_model_bundle(task)` (`@st.cache_resource` → `(tokenizer, model, label_map)` trên CPU, eval mode); `predict_feedback(text) -> dict` gọi `clean_text` + cả 2 model; trả `{sentiment_id, sentiment, sentiment_confidence, sentiment_probs, topic_id, topic, topic_confidence, topic_probs, text_clean, truncated}`. Kèm `models_available() -> dict` kiểm tra artifact tồn tại. |
| `src/dashboard/views/prediction.py` | View "Demo dự đoán": text_area, nút "Dự đoán", hiển thị 2 nhãn + confidence, bảng xác suất từng lớp, ghi chú `max_length=32` và disclaimer "xác suất softmax, không phải xác suất đã calibration". |

## 5. Files To Modify

| File | Sửa gì | Mức độ |
|---|---|---|
| `src/dashboard/app.py` | Thêm `"Demo dự đoán"` vào radio list + nhánh `elif page == "Demo dự đoán": prediction.render()` + import `prediction` | ~3 dòng, không đổi logic Phase 1 |
| `README.md` *(tuỳ chọn, mức thấp)* | Ghi thêm 1 dòng trong mục How to run: dashboard có tab demo inference cần `models/transformer/` local | 1–2 dòng; có thể để task sau |

`src/dashboard/loader.py`: **Không cần sửa** — inference tách module riêng, không trộn cache_data (data) với cache_resource (model).

## 6. Files That Must NOT Be Modified

- `data/raw/` và `data/processed/` — READ-ONLY tuyệt đối.
- `models/` — chỉ đọc, không ghi/xoá/sửa checkpoint; không commit vào Git.
- `src/preprocessing/*` — chỉ import `clean_text`, không sửa.
- `src/transformer/*` — chỉ tái sử dụng (import config/dataset nếu cần), không sửa.
- `src/baseline/*`, `src/eda/*` — không đụng.
- `results/*`, `reports/*` (trừ file plan này) — không ghi.
- `src/dashboard/loader.py`, `src/dashboard/views/*.py` Phase 1 — không sửa.
- `requirements.txt` — không thêm package.

## 7. Dependency Impact

**Không thêm package mới.** `torch>=2.4`, `transformers>=5.0`, `streamlit>=1.36`, `pandas` đều đã có trong `requirements.txt` và đã cài. `clean_text` thuần stdlib (`re`, `unicodedata`).

## 8. Model Loading Strategy

- `@st.cache_resource` cho `get_model_bundle(task)` — đúng loại cache cho object không serialize được (model/tokenizer). Load **1 lần/session/task**, không phải mỗi rerun.
- Lazy: chỉ load khi user mở tab Demo và bấm "Dự đoán" — tránh 1 GB RAM khi chỉ xem Phase 1.
- `models_available()` kiểm tra `models/transformer/{task}/model.safetensors` + `label_map.json` + `tokenizer_config.json` trước → thiếu thì hiển thị hướng dẫn lấy model thay vì crash.
- RAM estimate: 2 model × ~540 MB ≈ 1.1 GB — chấp nhận được cho laptop demo; có thể thêm checkbox "Chỉ dùng 1 task" nếu cần, nhưng mặc định load cả 2 khi predict.

## 9. Input Validation

| Trường hợp | Xử lý |
|---|---|
| Rỗng / chỉ whitespace | `st.warning("Vui lòng nhập phản hồi.")`, không gọi model |
| Quá dài | Cho phép; tokenizer `truncation=True` tại 32; hiển thị cờ `truncated` + ghi chú "văn bản dài hơn max_length=32 sẽ bị cắt" |
| Tiếng Việt có dấu | OK — `clean_text` NFC normalize |
| Emoji / acronym (colonsmile...) | OK — `clean_text` map acronym→Unicode giống training |
| Ký tự đặc biệt | OK — tokenizer xử lý; không strip gì thêm |
| Rất ngắn (1 từ) | Cho phép predict; không cảnh báo riêng (model đã train trên text ngắn) |
| Model/tokenizer/label_map thiếu | `st.error` + đường dẫn cần có, không fallback giả |
| `clean_text` trả "" sau xử lý | Coi như input rỗng → warning |

Không over-engineer: không rate-limit, không history, không batch.

## 10. Output Contract

`predict_feedback(raw_text) -> dict`:

```python
{
  "text_clean": str,            # sau clean_text
  "truncated": bool,            # len(tokens) > 32
  "sentiment": {"id": int, "label": str, "confidence": float,
                "probs": {label: float}},
  "topic":     {"id": int, "label": str, "confidence": float,
                "probs": {label: float}},
}
```

- `label` từ `label_map.json` checkpoint — domain: sentiment ∈ {Negative, Neutral, Positive}; topic ∈ {Lecturer, Training_program, Facility, Others}.
- `confidence` = softmax prob của lớp argmax — **được hỗ trợ an toàn** (đọc logits có sẵn). UI phải ghi "xác suất softmax của model" — không fabricate, không claim calibration (project chưa implement calibration — ghi trong limitation).

## 11. UI Plan

- Vị trí: entry **"Demo dự đoán"** trong sidebar radio, đặt sau "Phân tích lỗi", trước "Thời gian". Giữ nguyên 8 tab Phase 1.
- Layout: `st.text_area` (tiếng Việt, placeholder ví dụ) → nút `st.button("Dự đoán")` → 2 `st.metric`/columns cho Sentiment + Topic (label + confidence %) → `st.expander` "Chi tiết xác suất" bảng probs từng lớp → caption `text_clean` sau xử lý + cảnh báo truncate nếu có.
- Disclaimer cố định dưới tab: "Kết quả từ PhoBERT (MỨC 1). Xác suất là softmax của model, chưa calibration. Model chạy CPU — lần dự đoán đầu có thể mất vài giây."
- Nếu model thiếu: `st.info` hướng dẫn đặt `models/transformer/` (không crash Phase 1 — tab khác vẫn dùng được).

## 12. Validation Plan

1. **Import test** — `import src.dashboard.inference`, `src.dashboard.views.prediction` OK.
2. **Model loading test** — `get_model_bundle` cả 2 task load được; `cache_resource` chỉ load 1 lần (đếm số lần `from_pretrained`).
3. **Preprocessing consistency test** — `clean_text` trên mẫu từ `test.csv` cho ra đúng `text_clean` đã lưu (so khớp vài dòng) → chứng minh input contract giống training.
4. **Known-sample inference test** — lấy 3–5 mẫu test có `text`, predict → so `predicted_label` trong `*_test_predictions.csv`; kỳ vọng khớp nhãn (không bắt buộc khớp 100% — chỉ sanity).
5. **Vietnamese input test** — câu có dấu + emoji acronym (`colonsmile`) → ra label hợp lệ.
6. **Empty input test** — `""`, `"   "` → warning, không gọi model.
7. **Long input test** — văn bản >32 token → `truncated=True`, UI báo cắt.
8. **Streamlit smoke test** — `python -m streamlit run src/dashboard/app.py` headless → HTTP 200, click tab demo (render test như Phase 1).
9. **Regression test Phase 1** — chạy lại render() của 8 view cũ → không lỗi.
10. **Git diff inspection** — chỉ `app.py` (+~3 dòng), `inference.py`, `views/prediction.py`, `README.md` (nếu sửa); `git status` trên `data/`, `models/`, `results/` rỗng.

## 13. Acceptance Criteria

PASS nếu:

- Dashboard chạy `python -m streamlit run src/dashboard/app.py` không lỗi.
- Tab "Demo dự đoán" nhận input, bấm nút → ra sentiment + topic hợp lệ theo `label_map.json`.
- Confidence hiển thị là softmax thật từ logits (không số giả), kèm disclaimer.
- Input rỗng/whitespace được chặn có thông báo.
- Model load một lần qua `st.cache_resource` (không reload mỗi rerun).
- Cả 8 tab Phase 1 vẫn hoạt động.
- Không file nào trong `data/`, `models/`, `results/`, `reports/` (trừ plan) bị sửa.
- Không hard-code `D:\DeTai2_NLP`; mọi path qua `parents[2]`/`config.MODELS_DIR`.
- Không commit model/data; git status sạch ngoài file dashboard mới.

## 14. Risks / Blockers

| Rủi ro | Đánh giá | Giảm thiểu |
|---|---|---|
| RAM ~1.1 GB khi load cả 2 model | Thấp — laptop thường đủ | Lazy load, cache_resource; ghi chú trong UI |
| Cold start ~3 s mỗi model lần đầu | Thấp | Spinner "Đang tải model…", cache_resource |
| `config.id2label` generic trong checkpoint | Đã xử lý | Decode bằng `label_map.json`, không dùng config |
| Model chưa có trên máy mới (git-ignored) | Trung bình — môi trường deploy khác sẽ thiếu | `models_available()` check + hướng dẫn; README note |
| Preprocessing lệch train/inference | **Không có** — cùng `clean_text`, tokenizer mặc định = training | Test ở mục 12.3 |

**NO BLOCKER** cho việc implement trên máy hiện tại (model artifact đầy đủ local).
Rủi ro duy nhất thực chất: `models/` không trong Git → máy khác clone repo sẽ thiếu checkpoint; cần copy thủ công — ghi trong README note (mục 5).

## 15. Implementation Order

1. `src/dashboard/inference.py` — `models_available()`, `get_model_bundle(task)` với `@st.cache_resource`, `predict_feedback(text)` gọi `clean_text` + 2 model.
2. `src/dashboard/views/prediction.py` — UI tab Demo theo mục 11.
3. `src/dashboard/app.py` — import + radio entry + elif (minimal diff).
4. Validation theo mục 12 (import → model → consistency → sample → UI → regression → git diff).
5. *(Tuỳ chọn)* README note 1–2 dòng về tab demo + yêu cầu `models/transformer/` local.

## 16. Git Safety

- Phase audit này **không commit, không push**.
- Khi implement: không add `models/`, `data/raw/`, `*.safetensors`, cache, `.env` — `.gitignore` đã chặn; verify `git status` trước mọi commit.
- Không reset/rebase/cherry-pick lịch sử.
