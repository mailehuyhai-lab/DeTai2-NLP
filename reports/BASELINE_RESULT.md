# BASELINE RESULT

**Đề tài #2:** Phân tích cảm xúc và chủ đề từ phản hồi người dùng bằng NLP kết hợp Ontology cảm xúc

**Mức 1 — Baseline:** TF-IDF + Logistic Regression + Linear SVM

**Ngày thực hiện:** 2026-09-30

---

## 1. Environment

| Thành phần | Phiên bản |
|---|---|
| Python | 3.12.10 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| joblib | 1.6.0 |
| random_state | **42** (cố định cho reproducibility) |

---

## 2. Dataset

| Split | Samples | Nguồn |
|---|---|---|
| Train | 11,424 | `data/processed/train.csv` |
| Dev | 1,583 | `data/processed/dev.csv` |
| Test | 3,166 | `data/processed/test.csv` |
| **Tổng** | **16,173** | |

- **Text feature:** `text_lower` (lowercase, cho TF-IDF — theo Q4=C đã duyệt)
- **Sentiment labels:** 0=Negative, 1=Neutral, 2=Positive (3 lớp)
- **Topic labels:** 0=Lecturer, 1=Training_program, 2=Facility, 3=Others (4 lớp)
- **Class weighting:** `class_weight="balanced"` trong cả LR lẫn SVM

---

## 3. TF-IDF configuration

| Parameter | Giá trị | Lý do |
|---|---|---|
| `ngram_range` | (1, 2) | Unigrams + bigrams cho short feedback |
| `min_df` | 2 | Loại bỏ từ xuất hiện < 2 lần |
| `max_df` | 0.95 | Loại bỏ từ quá phổ biến (>95%) |
| `sublinear_tf` | True | 1 + log(tf) — tốt hơn tf thuần cho text classification |
| `max_features` | 50000 | Giới hạn vocabulary |
| `strip_accents` | None | Giữ nguyên dấu tiếng Việt |
| `lowercase` | False | `text_lower` đã lowercase sẵn |

**Vocabulary size sau fit:** 13,115 features

**TF-IDF chỉ fit trên TRAIN.** Dev và Test chỉ transform — không có data leakage.

---

## 4. Models

### 4.1 Logistic Regression

| Parameter | Giá trị |
|---|---|
| `solver` | `lbfgs` |
| `max_iter` | 1000 |
| `class_weight` | `balanced` |
| `random_state` | 42 |

### 4.2 Linear SVM (LinearSVC)

| Parameter | Giá trị |
|---|---|
| `max_iter` | 2000 |
| `class_weight` | `balanced` |
| `dual` | `auto` |
| `random_state` | 42 |

---

## 5. Sentiment results

### 5.1 Logistic Regression — Sentiment

**DEV:**

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 0.90 | 0.90 | 0.90 | 705 |
| Neutral | 0.35 | 0.59 | 0.44 | 73 |
| Positive | 0.96 | 0.89 | 0.92 | 805 |
| **Accuracy** | | | **0.88** | 1,583 |
| **Macro avg** | 0.73 | 0.79 | **0.75** | 1,583 |
| **Weighted avg** | 0.90 | 0.88 | **0.89** | 1,583 |

**TEST:**

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 0.87 | 0.91 | 0.89 | 1,409 |
| Neutral | 0.31 | 0.41 | 0.35 | 167 |
| Positive | 0.95 | 0.87 | 0.91 | 1,590 |
| **Accuracy** | | | **0.87** | 3,166 |
| **Macro avg** | 0.71 | 0.73 | **0.72** | 3,166 |
| **Weighted avg** | 0.88 | 0.87 | **0.87** | 3,166 |

### 5.2 Linear SVM — Sentiment

**DEV:**

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 0.91 | 0.93 | 0.92 | 705 |
| Neutral | 0.50 | 0.40 | 0.44 | 73 |
| Positive | 0.94 | 0.94 | 0.94 | 805 |
| **Accuracy** | | | **0.91** | 1,583 |
| **Macro avg** | 0.78 | 0.76 | **0.77** | 1,583 |
| **Weighted avg** | 0.91 | 0.91 | **0.91** | 1,583 |

**TEST:**

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 0.88 | 0.94 | 0.91 | 1,409 |
| Neutral | 0.45 | 0.30 | 0.36 | 167 |
| Positive | 0.93 | 0.91 | 0.92 | 1,590 |
| **Accuracy** | | | **0.89** | 3,166 |
| **Macro avg** | 0.75 | 0.72 | **0.73** | 3,166 |
| **Weighted avg** | 0.89 | 0.89 | **0.89** | 3,166 |

---

## 6. Topic results

### 6.1 Logistic Regression — Topic

**DEV:**

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Lecturer | 0.97 | 0.84 | 0.90 | 1,151 |
| Training_program | 0.61 | 0.86 | 0.71 | 267 |
| Facility | 0.84 | 0.91 | 0.88 | 70 |
| Others | 0.38 | 0.55 | 0.45 | 95 |
| **Accuracy** | | | **0.83** | 1,583 |
| **Macro avg** | 0.70 | 0.79 | **0.73** | 1,583 |
| **Weighted avg** | 0.87 | 0.83 | **0.84** | 1,583 |

**TEST:**

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Lecturer | 0.96 | 0.81 | 0.88 | 2,290 |
| Training_program | 0.59 | 0.80 | 0.68 | 572 |
| Facility | 0.90 | 0.92 | 0.91 | 145 |
| Others | 0.31 | 0.59 | 0.41 | 159 |
| **Accuracy** | | | **0.80** | 3,166 |
| **Macro avg** | 0.69 | 0.78 | **0.72** | 3,166 |
| **Weighted avg** | 0.86 | 0.80 | **0.82** | 3,166 |

### 6.2 Linear SVM — Topic

**DEV:**

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Lecturer | 0.94 | 0.91 | 0.92 | 1,151 |
| Training_program | 0.69 | 0.82 | 0.75 | 267 |
| Facility | 0.89 | 0.93 | 0.91 | 70 |
| Others | 0.51 | 0.43 | 0.47 | 95 |
| **Accuracy** | | | **0.87** | 1,583 |
| **Macro avg** | 0.76 | 0.77 | **0.76** | 1,583 |
| **Weighted avg** | 0.87 | 0.87 | **0.87** | 1,583 |

**TEST:**

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Lecturer | 0.93 | 0.91 | 0.92 | 2,290 |
| Training_program | 0.67 | 0.77 | 0.72 | 572 |
| Facility | 0.91 | 0.93 | 0.92 | 145 |
| Others | 0.48 | 0.43 | 0.46 | 159 |
| **Accuracy** | | | **0.86** | 3,166 |
| **Macro avg** | 0.75 | 0.76 | **0.75** | 3,166 |
| **Weighted avg** | 0.86 | 0.86 | **0.86** | 3,166 |

---

## 7. Model comparison

### Test set results (final evaluation)

| Task | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Sentiment | Logistic Regression | 0.8664 | 0.7174 | 0.8717 |
| Sentiment | **Linear SVM** | **0.8920** | **0.7304** | **0.8872** |
| Topic | Logistic Regression | 0.8042 | 0.7184 | 0.8213 |
| Topic | **Linear SVM** | **0.8588** | **0.7531** | **0.8602** |

### Quan sát từ metrics

**Sentiment:**
- Linear SVM đạt accuracy 89.2% và macro F1 0.73 trên test — cao hơn Logistic Regression (86.6%, macro F1 0.72).
- Lớp **Neutral** có F1 rất thấp ở cả hai model (LR: 0.35, SVM: 0.36) — do chỉ 167 mẫu test (5.3%) và tín hiệu ngữ nghĩa yếu.
- Lớp Negative và Positive có F1 cao (0.89–0.92).

**Topic:**
- Linear SVM đạt accuracy 85.9% và macro F1 0.75 — cao hơn Logistic Regression (80.4%, macro F1 0.72).
- Lớp **Others** có F1 thấp nhất (LR: 0.41, SVM: 0.46) — do ít mẫu (159 test, 5.0%) và nội dung đa dạng khó phân biệt.
- Lớp **Facility** đạt F1 cao nhất (0.91–0.92) dù ít mẫu — có lẽ do từ vựng đặc trưng rõ ràng.
- Lớp Lecturer chiếm 72% nhưng F1 vẫn cao (0.88–0.92).

---

## 8. Confusion matrices

### 8.1 Sentiment — Logistic Regression (Test)

| | Pred Negative | Pred Neutral | Pred Positive |
|---|---|---|---|
| **True Negative** | **1,289** | 85 | 35 |
| **True Neutral** | 63 | **69** | 35 |
| **True Positive** | 134 | 71 | **1,385** |

### 8.2 Sentiment — Linear SVM (Test)

| | Pred Negative | Pred Neutral | Pred Positive |
|---|---|---|---|
| **True Negative** | **1,331** | 29 | 49 |
| **True Neutral** | 63 | **50** | 54 |
| **True Positive** | 114 | 33 | **1,443** |

### 8.3 Topic — Logistic Regression (Test)

| | Pred Lecturer | Pred Training_program | Pred Facility | Pred Others |
|---|---|---|---|---|
| **True Lecturer** | **1,863** | 274 | 8 | 145 |
| **True Training_program** | 54 | **456** | 5 | 57 |
| **True Facility** | 3 | 4 | **133** | 5 |
| **True Others** | 21 | 42 | 2 | **94** |

### 8.4 Topic — Linear SVM (Test)

| | Pred Lecturer | Pred Training_program | Pred Facility | Pred Others |
|---|---|---|---|---|
| **True Lecturer** | **2,076** | 163 | 7 | 44 |
| **True Training_program** | 101 | **439** | 4 | 28 |
| **True Facility** | 2 | 6 | **135** | 2 |
| **True Others** | 42 | 45 | 3 | **69** |

---

## 9. Data leakage checks

| # | Check | Kết quả |
|---|---|---|
| 1 | TF-IDF fit chỉ trên TRAIN (11,424 samples) | ✅ PASS |
| 2 | Dev chỉ transform (1,583 samples) | ✅ PASS |
| 3 | Test chỉ transform (3,166 samples) | ✅ PASS |
| 4 | Prediction count = sample count (tất cả 4 models) | ✅ PASS |
| 5 | Không NaN trong metrics | ✅ PASS |
| 6 | Predictions trong valid label set | ✅ PASS |
| 7 | Confusion matrix kích thước đúng (sentiment 3×3, topic 4×4) | ✅ PASS |
| 8 | Saved models loadable và functional | ✅ PASS |
| 9 | Raw data MD5 checksums (9/9 files) | ✅ UNCHANGED |

---

## 10. Saved artifacts

### Models (`models/baseline/`)

| File | Kích thước | Mô tả |
|---|---|---|
| `tfidf_vectorizer.joblib` | 527 KB | TF-IDF vectorizer (13,115 features) |
| `sentiment_logistic_regression.joblib` | 316 KB | Sentiment — LR |
| `sentiment_linear_svm.joblib` | 316 KB | Sentiment — SVM |
| `topic_logistic_regression.joblib` | 421 KB | Topic — LR |
| `topic_linear_svm.joblib` | 421 KB | Topic — SVM |

### Results (`results/baseline/`)

| File | Mô tả |
|---|---|
| `baseline_metrics.json` | Tất cả metrics, config, parameters |
| `model_comparison.csv` | Bảng so sánh 4 models |
| `sentiment_classification_report.txt` | Classification report sentiment (dev+test) |
| `topic_classification_report.txt` | Classification report topic (dev+test) |
| `sentiment_logistic_regression_confusion_matrix.csv` | CM sentiment LR |
| `sentiment_linear_svm_confusion_matrix.csv` | CM sentiment SVM |
| `topic_logistic_regression_confusion_matrix.csv` | CM topic LR |
| `topic_linear_svm_confusion_matrix.csv` | CM topic SVM |
| `sentiment_logistic_regression_test_predictions.csv` | Predictions cho error analysis |
| `sentiment_linear_svm_test_predictions.csv` | Predictions cho error analysis |
| `topic_logistic_regression_test_predictions.csv` | Predictions cho error analysis |
| `topic_linear_svm_test_predictions.csv` | Predictions cho error analysis |

---

## 11. Conclusion

Baseline TF-IDF + Logistic Regression và TF-IDF + Linear SVM đã được xây dựng cho cả hai task (sentiment classification và topic classification) trên UIT-VSFC.

**Kết quả trên test set:**

- **Sentiment:** Linear SVM đạt accuracy 89.2%, macro F1 0.73, weighted F1 0.89. Lớp Neutral là thách thức lớn nhất (F1 = 0.36) do class imbalance nghiêm trọng (chỉ 5.3% test).

- **Topic:** Linear SVM đạt accuracy 85.9%, macro F1 0.75, weighted F1 0.86. Lớp Others là thách thức lớn nhất (F1 = 0.46) do ít mẫu và nội dung đa dạng.

- Cả hai task, **Linear SVM nhất quán đạt metrics cao hơn Logistic Regression** trên tất cả accuracy, macro F1, weighted F1.

- **Class imbalance ảnh hưởng rõ rệt:** Dù đã dùng `class_weight="balanced"`, các lớp thiểu số (Neutral, Others) vẫn có F1 thấp — đây là baseline kỳ vọng, Transformer/PhoBERT ở bước sau có thể cải thiện.

- Test predictions đã được lưu để phục vụ error analysis ở bước tiếp theo.

---

# BASELINE STATUS: ✅ READY FOR ERROR ANALYSIS AND TRANSFORMER

**DỪNG — Không tự chuyển sang PhoBERT hoặc bước tiếp theo.**
