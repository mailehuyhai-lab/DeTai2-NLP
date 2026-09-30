# PREPROCESSING RESULT

**Đề tài #2:** Phân tích cảm xúc và chủ đề từ phản hồi người dùng bằng NLP kết hợp Ontology cảm xúc

**Ngày thực hiện:** 2026-09-29

**Dựa trên:** PREPROCESSING_PLAN.md (đã duyệt Q1=A, Q2=A, Q3=A, Q4=C)

---

## 1. Files created

### Source code (`src/preprocessing/`)

| File | Mô tả | Dòng code |
|---|---|---|
| `__init__.py` | Package init | 2 |
| `load_raw.py` | Đọc & ghép raw data, MD5 checksum verification | ~100 |
| `text_cleaner.py` | Unicode NFC, emoji acronym→Unicode, whitespace norm | ~100 |
| `build_processed.py` | Pipeline chính: load → clean → export | ~250 |
| `validate.py` | 11 integrity + 5 quality checks | ~270 |

### Output data (`data/processed/`)

| File | Kích thước | Mô tả |
|---|---|---|
| `train.csv` | 2,965,145 bytes | 11,424 mẫu (sau xóa 2 conflict) |
| `dev.csv` | 390,571 bytes | 1,583 mẫu |
| `test.csv` | 815,371 bytes | 3,166 mẫu |
| `label_maps.json` | 204 bytes | Mapping nhãn int → text |
| `class_weights.json` | 410 bytes | Trọng số inverse-frequency |
| `preprocessing_log.json` | 2,793 bytes | Log chi tiết 8 bước xử lý |
| `stats_summary.json` | 1,124 bytes | Thống kê sau xử lý |

---

## 2. Sample counts

| Split | Raw (trước) | Processed (sau) | Thay đổi |
|---|---|---|---|
| train | 11,426 | **11,424** | -2 (xóa annotation conflict) |
| dev | 1,583 | **1,583** | 0 |
| test | 3,166 | **3,166** | 0 |
| **Tổng** | **16,175** | **16,173** | **-2** |

---

## 3. Rows removed

| # | Split | ID | Index gốc | Text | Sentiment | Topic | Lý do |
|---|---|---|---|---|---|---|---|
| 1 | train | `train_11293` | 11293 | `thầy dạy hay , tuy nhiên còn nhiều chỗ chưa thật sự giải đáp hoàn toàn cho sinh viên vì chưa đủ thời gian .` | 2 (Positive) | 0 (Lecturer) | Annotation conflict (Q1=A) |
| 2 | train | `train_11417` | 11417 | *(giống hệt)* | 0 (Negative) | 0 (Lecturer) | Annotation conflict (Q1=A) |

- Chỉ xóa trong bản processed. **`data/raw/` KHÔNG bị sửa.**
- Ghi đầy đủ trong `preprocessing_log.json`.

---

## 4. Text transformations

| Bước | Mô tả | Quyết định | Câu bị ảnh hưởng |
|---|---|---|---|
| 1 | Unicode NFC normalization | Mặc định | (gộp trong tổng bên dưới) |
| 2 | Emoji acronym → Emoji Unicode | Q2=A đã duyệt | 210 câu có acronym |
| 3 | Whitespace normalization (collapse + strip) | Mặc định | (gộp trong tổng bên dưới) |
| 4 | Punctuation spacing: giữ nguyên | Q3=A đã duyệt | 0 (không thay đổi) |
| 5 | Tạo `text_lower` = `text_clean.lower()` | Q4=C đã duyệt | Tất cả |

**Tổng câu có `text ≠ text_clean`:**

| Split | Số câu thay đổi | % |
|---|---|---|
| train | 200 | 1.75% |
| dev | 20 | 1.26% |
| test | 72 | 2.27% |
| **Tổng** | **292** | **1.80%** |

---

## 5. Label distributions

### Sentiment (sau xử lý)

| Lớp | Train | Dev | Test | Tổng |
|---|---|---|---|---|
| 0 — Negative | 5,324 (46.6%) | 705 (44.5%) | 1,409 (44.5%) | 7,438 |
| 1 — Neutral | 458 (4.0%) | 73 (4.6%) | 167 (5.3%) | 698 |
| 2 — Positive | 5,642 (49.4%) | 805 (50.9%) | 1,590 (50.2%) | 8,037 |

### Topic (sau xử lý)

| Lớp | Train | Dev | Test | Tổng |
|---|---|---|---|---|
| 0 — Lecturer | 8,164 (71.5%) | 1,151 (72.7%) | 2,290 (72.3%) | 11,605 |
| 1 — Training_program | 2,201 (19.3%) | 267 (16.9%) | 572 (18.1%) | 3,040 |
| 2 — Facility | 497 (4.3%) | 70 (4.4%) | 145 (4.6%) | 712 |
| 3 — Others | 562 (4.9%) | 95 (6.0%) | 159 (5.0%) | 816 |

---

## 6. Generated metadata

### `label_maps.json`
```json
{
  "sentiment": {"0": "Negative", "1": "Neutral", "2": "Positive"},
  "topic": {"0": "Lecturer", "1": "Training_program", "2": "Facility", "3": "Others"}
}
```

### `class_weights.json`
```json
{
  "sentiment_weights": {"0": 0.7153, "1": 8.3144, "2": 0.6749},
  "topic_weights": {"0": 0.3498, "1": 1.2976, "2": 5.7465, "3": 5.0819},
  "method": "inverse_frequency",
  "formula": "weight = total_samples / (num_classes * count_per_class)",
  "train_total": 11424
}
```

### `stats_summary.json`
- Tổng mẫu: train=11,424, dev=1,583, test=3,166 (total=16,173)
- Phân bố sentiment & topic chi tiết cho mỗi split
- Text length: train min=4, max=660, mean=58.9, median=47.0
- Conflicts removed: 2
- Emoji acronyms affected: train=200, dev=20, test=72

### `preprocessing_log.json`
- 8 bước xử lý ghi chi tiết
- Quyết định đã duyệt: Q1=A, Q2=A, Q3=A, Q4=C
- Chi tiết 2 dòng conflict bị xóa (text, nhãn, index)

---

## 7. Integrity checks

| # | Check | Kết quả |
|---|---|---|
| I-1 | Raw files unchanged (MD5 checksums) | ✅ PASS — 9/9 khớp |
| I-2 | Sample counts correct | ✅ PASS — train=11424, dev=1583, test=3166 |
| I-3 | No null/NaN values | ✅ PASS — 0 null |
| I-4 | No empty text strings | ✅ PASS — 0 empty |
| I-5 | Labels valid | ✅ PASS — sentiment∈{0,1,2}, topic∈{0,1,2,3} |
| I-6 | Unique IDs per file | ✅ PASS |
| I-7 | No cross-split text overlap | ✅ PASS — 0 overlapping |
| I-8 | UTF-8 encoding correct | ✅ PASS |
| I-9 | Dev/Test sample count unchanged | ✅ PASS — dev=1583, test=3166 |
| I-10 | Annotation conflict removed | ✅ PASS — conflict text not in train |
| I-11 | Dev/Test label distribution unchanged | ✅ PASS — match raw |

**Integrity: 11/11 PASSED ✅**

---

## 8. Quality checks

| # | Check | Kết quả |
|---|---|---|
| Q-1 | No emoji acronyms remaining | ✅ PASS — 0 acronyms in text_clean |
| Q-2 | No double spaces | ✅ PASS — 0 occurrences |
| Q-3 | text_lower == text_clean.lower() | ✅ PASS — 100% consistent |
| Q-4 | Text changed by preprocessing | ✅ PASS — 292 sentences changed |
| Q-5 | Emoji Unicode present | ✅ PASS — 110 emoji occurrences found |

**Quality: 5/5 PASSED ✅**

---

## 9. Raw MD5 verification

| File | MD5 (trước) | MD5 (sau) | Khớp |
|---|---|---|---|
| `train/sents.txt` | `dd4d13bee582f9120f5dcfa5d126662e` | `dd4d13bee582f9120f5dcfa5d126662e` | ✅ |
| `train/sentiments.txt` | `b807001a6c4c573cdd3fe236fa06add4` | `b807001a6c4c573cdd3fe236fa06add4` | ✅ |
| `train/topics.txt` | `337e2f21d16e51b3ee4d1013e018458c` | `337e2f21d16e51b3ee4d1013e018458c` | ✅ |
| `dev/sents.txt` | `4b01b2b3df63d6dd0c64c0e9c1f4c763` | `4b01b2b3df63d6dd0c64c0e9c1f4c763` | ✅ |
| `dev/sentiments.txt` | `bb87e5ee63082ea3c438f130a2a227aa` | `bb87e5ee63082ea3c438f130a2a227aa` | ✅ |
| `dev/topics.txt` | `11e9e0ea3beb1aafbdddc424b0696407` | `11e9e0ea3beb1aafbdddc424b0696407` | ✅ |
| `test/sents.txt` | `800f706107d9e634563848a681566c34` | `800f706107d9e634563848a681566c34` | ✅ |
| `test/sentiments.txt` | `e125fa119d7c1c0a21337a9e80b70f29` | `e125fa119d7c1c0a21337a9e80b70f29` | ✅ |
| `test/topics.txt` | `224c091149ad257ab72f742cf545ab06` | `224c091149ad257ab72f742cf545ab06` | ✅ |

**9/9 raw files KHÔNG bị thay đổi ✅**

---

## 10. Final status

```
============================================================
  Integrity: 11/11 PASSED ✅
  Quality:    5/5 PASSED ✅
  Raw data:   9/9 UNCHANGED ✅
============================================================
```

# PREPROCESSING STATUS: ✅ READY FOR MODEL TRAINING

Tất cả bước preprocessing đã hoàn thành đúng theo PREPROCESSING_PLAN.md đã duyệt.
Dữ liệu gốc `data/raw/` hoàn toàn không bị thay đổi.
Dữ liệu đã xử lý nằm tại `data/processed/`.

**DỪNG — Không tự chuyển sang bước train model.**
