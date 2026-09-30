# PREPROCESSING PLAN (FINAL — ĐÃ DUYỆT)

**Đề tài #2:** Phân tích cảm xúc và chủ đề từ phản hồi người dùng bằng NLP kết hợp Ontology cảm xúc

**Dựa trên:** DATASET_AUDIT.md (2026-09-29)

**Ngày lập:** 2026-09-29  
**Ngày duyệt:** 2026-09-29

**Trạng thái: ✅ ĐÃ DUYỆT — Sẵn sàng code preprocessing**

---

## Quyết định đã được duyệt

| # | Quyết định | Phương án được duyệt |
|---|---|---|
| **Q1** | Annotation conflict | **A. Xóa cả 2 dòng** khỏi bản processed. Không sửa `data/raw/` |
| **Q2** | Emoji acronym | **A. Chuyển về Emoji Unicode** (ví dụ: `colonsmile` → 🙂). Không xóa tín hiệu cảm xúc |
| **Q3** | Punctuation spacing | **A. Giữ nguyên** format pre-tokenized gốc UIT-VSFC |
| **Q4** | Lowercase | **C. Tạo 2 phiên bản**: `text_clean` (giữ case, cho PhoBERT) + `text_lower` (lowercase, cho TF-IDF) |

---

## Nguyên tắc tuyệt đối

- ❌ KHÔNG sửa bất kỳ file nào trong `data/raw/`
- ❌ KHÔNG xóa dữ liệu gốc
- ❌ KHÔNG tạo dữ liệu giả
- ❌ KHÔNG train model ở bước này
- ❌ KHÔNG làm dashboard/API ở bước này
- ✅ Mọi output nằm ngoài `data/raw/`

---

## 1. Proposed directory structure

```
DeTai2_NLP/
├── data/
│   ├── raw/                          # ❌ KHÔNG ĐƯỢC CHỈNH SỬA (read-only)
│   │   ├── README.txt
│   │   ├── train/  (sents.txt, sentiments.txt, topics.txt)
│   │   ├── dev/    (sents.txt, sentiments.txt, topics.txt)
│   │   └── test/   (sents.txt, sentiments.txt, topics.txt)
│   │
│   └── processed/                    # ✅ OUTPUT của preprocessing
│       ├── train.csv                 # 11,424 mẫu (sau xóa 2 conflict)
│       ├── dev.csv                   # 1,583 mẫu (không đổi)
│       ├── test.csv                  # 3,166 mẫu (không đổi)
│       ├── label_maps.json           # Mapping nhãn sentiment & topic
│       ├── class_weights.json        # Trọng số cho class imbalance
│       ├── preprocessing_log.json    # Log mọi thay đổi so với raw
│       └── stats_summary.json        # Thống kê sau xử lý
│
├── src/
│   └── preprocessing/
│       ├── __init__.py
│       ├── load_raw.py               # Đọc và ghép 3 file text → DataFrame
│       ├── text_cleaner.py           # Unicode norm, emoji acronym, whitespace
│       ├── build_processed.py        # Pipeline chính: load → clean → export
│       └── validate.py               # Kiểm tra output vs raw (integrity check)
│
├── reports/
│   ├── DATASET_AUDIT.md              # Đã hoàn thành
│   └── PREPROCESSING_PLAN.md         # File này
│
├── reference/
│   └── De_cuong_de_tai_CuChi_2026.docx
│
└── project/
```

---

## 2. Data loading strategy

### 2.1 Đọc raw data

Với mỗi split (`train`, `dev`, `test`):

1. Đọc `sents.txt` (encoding UTF-8), mỗi dòng → 1 câu.
2. Đọc `sentiments.txt` (encoding ASCII), mỗi dòng → 1 nhãn sentiment (int).
3. Đọc `topics.txt` (encoding ASCII), mỗi dòng → 1 nhãn topic (int).
4. **Assert** 3 file có cùng số dòng.
5. Ghép theo index dòng → `pandas.DataFrame`.

### 2.2 Schema sau khi ghép (trước preprocessing)

| Cột | Kiểu | Nguồn |
|---|---|---|
| `id` | string | Tự sinh: `{split}_{index:05d}` (ví dụ: `train_00000`) |
| `text` | string | `sents.txt` dòng tương ứng, `.strip()` |
| `sentiment` | int | `sentiments.txt` dòng tương ứng |
| `topic` | int | `topics.txt` dòng tương ứng |
| `split` | string | `"train"` / `"dev"` / `"test"` |

### 2.3 Nguyên tắc

- Đọc **read-only** từ `data/raw/`.
- Dùng `open()` + `readlines()` rồi ghép (không dùng `pd.read_csv` vì file không có header/separator).
- **Tính checksum MD5** của 9 file raw trước khi đọc và sau khi xử lý xong, để xác nhận raw không bị sửa.

### 2.4 Checksums tham chiếu (raw data — phải giữ nguyên)

```
dd4d13bee582f9120f5dcfa5d126662e  train/sents.txt
b807001a6c4c573cdd3fe236fa06add4  train/sentiments.txt
337e2f21d16e51b3ee4d1013e018458c  train/topics.txt
4b01b2b3df63d6dd0c64c0e9c1f4c763  dev/sents.txt
bb87e5ee63082ea3c438f130a2a227aa  dev/sentiments.txt
11e9e0ea3beb1aafbdddc424b0696407  dev/topics.txt
800f706107d9e634563848a681566c34  test/sents.txt
e125fa119d7c1c0a21337a9e80b70f29  test/sentiments.txt
224c091149ad257ab72f742cf545ab06  test/topics.txt
```

---

## 3. Label mapping

### 3.1 Sentiment labels

| Giá trị (int) | Nhãn text | Mô tả |
|---|---|---|
| `0` | `"Negative"` | Tiêu cực |
| `1` | `"Neutral"` | Trung lập |
| `2` | `"Positive"` | Tích cực |

### 3.2 Topic labels

| Giá trị (int) | Nhãn text | Mô tả |
|---|---|---|
| `0` | `"Lecturer"` | Giảng viên |
| `1` | `"Training_program"` | Chương trình đào tạo |
| `2` | `"Facility"` | Cơ sở vật chất |
| `3` | `"Others"` | Khác |

### 3.3 Lưu trữ → `data/processed/label_maps.json`

```json
{
  "sentiment": {"0": "Negative", "1": "Neutral", "2": "Positive"},
  "topic": {"0": "Lecturer", "1": "Training_program", "2": "Facility", "3": "Others"}
}
```

Cột `sentiment` và `topic` trong CSV giữ **giá trị int gốc**. Nhãn text chỉ dùng cho hiển thị/báo cáo.

---

## 4. Text preprocessing

### 4.1 Pipeline (tuần tự)

```
text (gốc từ sents.txt, chỉ strip)
  │
  ├─ Bước 1: Unicode normalization (NFC)
  ├─ Bước 2: Emoji acronym → Emoji Unicode       ← Q2 đã duyệt
  ├─ Bước 3: Chuẩn hóa khoảng trắng
  ├─ Bước 4: Punctuation spacing → Giữ nguyên    ← Q3 đã duyệt
  │
  ▼
text_clean (giữ case gốc, dùng cho PhoBERT)      ← Q4 đã duyệt
  │
  ├─ Bước 5: Lowercase
  │
  ▼
text_lower (lowercase, dùng cho TF-IDF baseline)  ← Q4 đã duyệt
```

Cột `text` gốc được giữ lại trong CSV để đối chiếu.

### 4.2 Bước 1: Unicode normalization

- `unicodedata.normalize('NFC', text)`
- Đảm bảo ký tự tiếng Việt nhất quán (ví dụ: `ă` = 1 codepoint).
- Đề cương yêu cầu: "Chuẩn hóa Unicode".

### 4.3 Bước 2: Emoji acronym → Emoji Unicode (Q2=A ĐÃ DUYỆT)

**Bảng chuyển đổi chính thức:**

| Acronym | → Emoji/Ký tự | Unicode | Ghi chú |
|---|---|---|---|
| `colonsmile` | 🙂 | U+1F642 | Cười |
| `colonsad` | 😞 | U+1F61E | Buồn |
| `colonsurprise` | 😮 | U+1F62E | Ngạc nhiên |
| `colonlove` | ❤️ | U+2764 | Yêu |
| `colonsmilesmile` | 😄 | U+1F604 | Cười to |
| `coloncontemn` | 😏 | U+1F60F | Khinh / dễ thương |
| `colonbigsmile` | 😆 | U+1F606 | Cười lớn |
| `coloncc` | 😢 | U+1F622 | Khóc |
| `colonsmallsmile` | 😋 | U+1F60B | Lè lưỡi |
| `coloncolon` | `>>` | — | Giữ nguyên (không phải emoji) |
| `colonlovelove` | 🥰 | U+1F970 | Yêu thương |
| `colonhihi` | 😊 | U+1F60A | Vui vẻ |
| `doubledot` | `:` | U+003A | Dấu hai chấm |
| `colonsadcolon` | 😢 | U+1F622 | Khóc |
| `colondoublesurprise` | 😡 | U+1F621 | Tức giận |
| `vdotv` | `v.v` | — | Giữ nguyên (viết tắt "vân vân") |
| `dotdotdot` | `...` | — | Ba dấu chấm |
| `fraction` | `/` | U+002F | Dấu gạch chéo |
| `cshrap` | `c#` | — | Ngôn ngữ lập trình |

**Lưu ý thứ tự thay thế:** Thay acronym dài trước, ngắn sau, để tránh xung đột (ví dụ: `colonsmilesmile` trước `colonsmile`, `colonsadcolon` trước `colonsad`, `colondoublesurprise` trước `colonsurprise`).

**Số câu bị ảnh hưởng:** 210 câu (train: 148, dev: 11, test: 51).

### 4.4 Bước 3: Chuẩn hóa khoảng trắng

- `re.sub(r'\s+', ' ', text).strip()`
- Thay nhiều khoảng trắng liên tiếp → 1 khoảng trắng.
- Xóa khoảng trắng đầu/cuối câu.

### 4.5 Bước 4: Punctuation spacing (Q3=A ĐÃ DUYỆT)

- **Giữ nguyên** format pre-tokenized (ví dụ: `"dễ ."` vẫn giữ khoảng trắng trước dấu chấm).
- Lý do: Đây là format chuẩn UIT-VSFC, PhoBERT tokenizer xử lý tốt format này.
- **Không cần code** cho bước này — chỉ ghi nhận quyết định.

### 4.6 Bước 5: Lowercase (Q4=C ĐÃ DUYỆT)

- `text_lower = text_clean.lower()`
- Tạo cột riêng `text_lower` cho TF-IDF baseline.
- Cột `text_clean` giữ nguyên case cho PhoBERT.

### 4.7 Không xử lý (giữ nguyên)

| Loại | Quyết định | Lý do |
|---|---|---|
| Câu rất ngắn (4-5 ký tự) | Giữ nguyên | Dữ liệu thực, mang tín hiệu sentiment rõ |
| Câu tiếng Anh (~21 câu) | Giữ nguyên | Số lượng không đáng kể, PhoBERT xử lý được |

---

## 5. Duplicate handling

- **1 cặp trùng lặp** trong train (index 11293 và 11417) — có annotation conflict.
- Xử lý: **Xóa cả 2** (xem mục 6).
- Dev, test: không có trùng lặp.
- Cross-split: không có trùng lặp.

---

## 6. Annotation conflict handling (Q1=A ĐÃ DUYỆT)

### 6.1 Chi tiết conflict

| Thuộc tính | Dòng index 11293 | Dòng index 11417 |
|---|---|---|
| **Text** | `thầy dạy hay , tuy nhiên còn nhiều chỗ chưa thật sự giải đáp hoàn toàn cho sinh viên vì chưa đủ thời gian .` | *(giống hệt)* |
| **Sentiment** | **2 (Positive)** | **0 (Negative)** |
| **Topic** | 0 (Lecturer) | 0 (Lecturer) |

### 6.2 Quyết định: XÓA CẢ 2 DÒNG khỏi bản processed

- Chỉ mất 2/11,426 mẫu (0.017%) — không ảnh hưởng phân bố.
- Không đưa ra quyết định chủ quan về nhãn.
- Ghi đầy đủ vào `preprocessing_log.json` với text gốc, nhãn, index, lý do.
- **Tuyệt đối không sửa `data/raw/`** — chỉ loại bỏ trong bản processed.

### 6.3 Ảnh hưởng đến phân bố

| | Trước xóa | Sau xóa | Thay đổi |
|---|---|---|---|
| Train total | 11,426 | **11,424** | -2 |
| Sentiment 0 (Neg) | 5,325 | **5,324** | -1 |
| Sentiment 2 (Pos) | 5,643 | **5,642** | -1 |
| Topic 0 (Lecturer) | 8,166 | **8,164** | -2 |

Sentiment 1 (Neutral), Topic 1/2/3: không thay đổi.  
Dev, Test: không thay đổi.

---

## 7. Class imbalance strategy

### 7.1 Phân bố sau xử lý conflict

**Sentiment (train, n=11,424):**
| Lớp | Count | % | Weight |
|---|---|---|---|
| Negative (0) | 5,324 | 46.6% | **0.7153** |
| Neutral (1) | 458 | 4.0% ⚠️ | **8.3144** |
| Positive (2) | 5,642 | 49.4% | **0.6749** |

**Topic (train, n=11,424):**
| Lớp | Count | % | Weight |
|---|---|---|---|
| Lecturer (0) | 8,164 | 71.5% ⚠️ | **0.3498** |
| Training_program (1) | 2,201 | 19.3% | **1.2976** |
| Facility (2) | 497 | 4.3% ⚠️ | **5.7465** |
| Others (3) | 562 | 4.9% ⚠️ | **5.0819** |

### 7.2 Công thức

```
weight[class_i] = total_samples / (num_classes × count[class_i])
```

Tương đương `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`.

### 7.3 Lưu trữ → `data/processed/class_weights.json`

```json
{
  "sentiment_weights": {"0": 0.7153, "1": 8.3144, "2": 0.6749},
  "topic_weights": {"0": 0.3498, "1": 1.2976, "2": 5.7465, "3": 5.0819},
  "method": "inverse_frequency",
  "formula": "weight = total_samples / (num_classes * count_per_class)",
  "train_total": 11424,
  "note": "Chỉ tính trên train set. Áp dụng khi train model, KHÔNG sửa data."
}
```

### 7.4 Chiến lược bổ sung (ghi nhận cho bước training — CHƯA áp dụng)

| Chiến lược | Mô tả | Khi nào dùng |
|---|---|---|
| **Weighted loss** | Dùng class_weight trong loss function | Ưu tiên số 1 cho cả sentiment lẫn topic |
| **Focal loss** | γ=2.0 mặc định, giảm ảnh hưởng easy examples | Nếu weighted loss chưa đủ (neutral chỉ 4%) |
| **Oversampling** | Random oversample lớp thiểu số trên train | Xem xét nếu cần, chỉ trên train |
| **Stratified batching** | Đảm bảo mỗi batch có đủ lớp thiểu số | Cho deep learning |

**Quan trọng:** Dev và test KHÔNG ĐƯỢC augment/resample — giữ nguyên phân bố gốc.

---

## 8. Output schema

### 8.1 File CSV (`train.csv`, `dev.csv`, `test.csv`)

| Cột | Kiểu | Mô tả | Ví dụ |
|---|---|---|---|
| `id` | string | ID duy nhất: `{split}_{index:05d}` | `train_00000` |
| `text` | string | Văn bản gốc (từ `sents.txt`, chỉ `.strip()`) | `slide giáo trình đầy đủ .` |
| `text_clean` | string | Sau NFC + emoji→Unicode + whitespace norm (giữ case) | `slide giáo trình đầy đủ .` |
| `text_lower` | string | `text_clean.lower()` (cho TF-IDF) | `slide giáo trình đầy đủ .` |
| `sentiment` | int | Nhãn cảm xúc: 0, 1, 2 | `2` |
| `topic` | int | Nhãn chủ đề: 0, 1, 2, 3 | `1` |

### 8.2 Encoding & Format

| Thuộc tính | Giá trị |
|---|---|
| Encoding | UTF-8 (không BOM) |
| Separator | `,` (comma) |
| Quoting | `csv.QUOTE_NONNUMERIC` |
| Header | Có (dòng đầu = tên cột) |
| Line ending | LF (`\n`) |
| Pandas index | Không lưu (`index=False`) |

### 8.3 Số mẫu kỳ vọng

| Split | Raw | Processed | Thay đổi |
|---|---|---|---|
| train | 11,426 | **11,424** | -2 (xóa conflict) |
| dev | 1,583 | **1,583** | 0 |
| test | 3,166 | **3,166** | 0 |
| **Tổng** | **16,175** | **16,173** | **-2** |

### 8.4 File phụ trợ

| File | Nội dung |
|---|---|
| `label_maps.json` | Mapping int → tên nhãn |
| `class_weights.json` | Trọng số inverse-frequency |
| `preprocessing_log.json` | Log chi tiết mọi bước, mọi thay đổi |
| `stats_summary.json` | Thống kê cuối cùng sau xử lý |

---

## 9. Validation checks

Script `validate.py` chạy sau preprocessing, phải **pass tất cả** trước khi tiến hành bước tiếp theo.

### 9.1 Integrity checks (PHẢI PASS — fail = dừng lại)

| # | Check | Tiêu chí | Trạng thái |
|---|---|---|---|
| I-1 | **Raw không bị sửa** | MD5 checksum 9 file raw = giá trị tham chiếu (mục 2.4) | ☐ |
| I-2 | **Số mẫu đúng** | train=11,424, dev=1,583, test=3,166 | ☐ |
| I-3 | **Không có null/NaN** | Tất cả 6 cột trong mỗi CSV: 0 null | ☐ |
| I-4 | **Không có dòng trống** | `text`, `text_clean`, `text_lower`: không có chuỗi rỗng | ☐ |
| I-5 | **Nhãn hợp lệ** | sentiment ∈ {0,1,2}, topic ∈ {0,1,2,3} cho mọi dòng | ☐ |
| I-6 | **ID duy nhất** | Không trùng `id` trong mỗi file | ☐ |
| I-7 | **Cross-split clean** | Không có `text` trùng giữa train/dev/test | ☐ |
| I-8 | **Encoding đúng** | Tất cả CSV đọc được với `encoding='utf-8'` | ☐ |
| I-9 | **Dev/Test không đổi số mẫu** | So sánh count dòng raw vs processed | ☐ |
| I-10 | **Conflict đã xóa** | Câu conflict không tồn tại trong train.csv processed | ☐ |
| I-11 | **Phân bố dev/test giữ nguyên** | Phân bố nhãn dev/test processed = raw | ☐ |

### 9.2 Quality checks (CẢNH BÁO — không fail nhưng phải báo cáo)

| # | Check | Tiêu chí | Trạng thái |
|---|---|---|---|
| Q-1 | **Không còn emoji acronym** | 0 lần xuất hiện `colonsmile`, `colonsad`,... trong `text_clean` | ☐ |
| Q-2 | **Không còn khoảng trắng thừa** | 0 dòng có `"  "` (2+ spaces) trong `text_clean` | ☐ |
| Q-3 | **text_lower nhất quán** | `text_lower == text_clean.lower()` cho 100% dòng | ☐ |
| Q-4 | **text vs text_clean khác nhau** | Đếm số dòng text ≠ text_clean (= ảnh hưởng của NFC + emoji) | ☐ |
| Q-5 | **Emoji Unicode hiện diện** | Kiểm tra 🙂😞❤️ v.v. xuất hiện đúng số lần trong text_clean | ☐ |

### 9.3 Output của validate.py

```
=== INTEGRITY CHECKS ===
[PASS] I-1: Raw files unchanged (9/9 checksums match)
[PASS] I-2: Sample counts correct (train=11424, dev=1583, test=3166)
...
=== QUALITY CHECKS ===
[PASS] Q-1: No emoji acronyms remaining (0 found)
...
=== RESULT ===
Integrity: 11/11 PASSED
Quality:   5/5 PASSED
Status:    ✅ READY FOR TRAINING
```

---

## 10. Tổng kết — Sẵn sàng code

### Pipeline preprocessing tóm tắt

```
data/raw/ (READ-ONLY)
    │
    ├─ 1. Load: sents.txt + sentiments.txt + topics.txt → DataFrame
    ├─ 2. Merge: ghép theo index dòng, thêm id + split
    ├─ 3. Remove conflict: xóa 2 dòng index 11293 & 11417 (chỉ train)
    ├─ 4. Text clean:
    │     ├─ NFC normalize
    │     ├─ Emoji acronym → Emoji Unicode
    │     └─ Whitespace normalize
    ├─ 5. Generate text_lower: text_clean.lower()
    ├─ 6. Export: train.csv, dev.csv, test.csv
    ├─ 7. Export metadata: label_maps.json, class_weights.json
    ├─ 8. Export logs: preprocessing_log.json, stats_summary.json
    └─ 9. Validate: integrity + quality checks
    │
    ▼
data/processed/ (OUTPUT)
```

### Checklist trước khi bắt đầu code

| # | Điều kiện | Trạng thái |
|---|---|---|
| 1 | DATASET_AUDIT.md hoàn thành | ✅ |
| 2 | PREPROCESSING_PLAN.md hoàn thành | ✅ |
| 3 | Q1 (Annotation conflict) đã duyệt: Xóa cả 2 | ✅ |
| 4 | Q2 (Emoji acronym) đã duyệt: Chuyển về Emoji Unicode | ✅ |
| 5 | Q3 (Punctuation spacing) đã duyệt: Giữ nguyên | ✅ |
| 6 | Q4 (Lowercase) đã duyệt: 2 phiên bản | ✅ |
| 7 | MD5 checksums raw data đã ghi nhận | ✅ |
| 8 | Class weights đã tính chính xác | ✅ |
| 9 | Validation checklist đã xác định | ✅ |
| 10 | Chưa train model | ✅ |
| 11 | Chưa làm dashboard/API | ✅ |
| 12 | data/raw/ chưa bị sửa | ✅ |

**→ Sẵn sàng chờ duyệt bước CODE PREPROCESSING.**
