# ERROR ANALYSIS RESULT

**Đề tài #2:** Phân tích cảm xúc và chủ đề từ phản hồi người dùng bằng NLP kết hợp Ontology cảm xúc

**Mức 1 — Error Analysis:** Baseline Linear SVM trên UIT-VSFC test set

**Ngày thực hiện:** 2026-09-30

---

## 1. Objective

Phân tích các mẫu mà Linear SVM (baseline tốt nhất) dự đoán sai trên test set, nhằm:

- Xác định phân bố lỗi theo từng cặp (true → predicted).
- Nhận diện các error pattern từ dữ liệu thực tế.
- Rút ra hàm ý cho bước Transformer tiếp theo.

Chỉ phân tích predictions đã lưu trong `results/baseline/` — không re-train, không tạo prediction mới.

---

## 2. Models analyzed

| Task | Model | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---|---|---|---|
| Sentiment | Linear SVM | 0.8920 | 0.7304 | 0.8872 |
| Topic | Linear SVM | 0.8588 | 0.7531 | 0.8602 |

Đối chiếu chéo với Logistic Regression để phân biệt lỗi model-specific vs. hard cases (xem section 5).

**Input files:**
- `results/baseline/sentiment_linear_svm_test_predictions.csv` (3,166 rows, schema: id, text, true_label, predicted_label, correct, true_label_name, predicted_label_name)
- `results/baseline/topic_linear_svm_test_predictions.csv` (cùng schema)
- `results/baseline/sentiment_logistic_regression_test_predictions.csv` (đối chiếu)
- `results/baseline/topic_logistic_regression_test_predictions.csv` (đối chiếu)

Prediction files không chứa decision score/confidence → phân tích không có thông tin confidence.

---

## 3. Sentiment error analysis

### 3.1 Error distribution

- Tổng test: 3,166 mẫu
- Số lỗi: **342** (10.8%)
- Số đúng: 2,824 (89.2%)

### 3.2 Main error pairs

| True → Predicted | Count | % of errors | % of true class |
|---|---|---|---|
| Positive → Negative | 114 | 33.33% | 7.17% |
| Neutral → Negative | 63 | 18.42% | 37.72% |
| Neutral → Positive | 54 | 15.79% | 32.34% |
| Negative → Positive | 49 | 14.33% | 3.48% |
| Positive → Neutral | 33 | 9.65% | 2.08% |
| Negative → Neutral | 29 | 8.48% | 2.06% |

**Quan sát chính:**

1. **Positive → Negative là lỗi lớn nhất** (114 mẫu, 33% lỗi). Nhiều mẫu Positive chứa từ negation (`không`, `chưa`) hoặc mô tả điều tốt trong bối cảnh thiếu/chưa có — TF-IDF n-gram không nắm được scope phủ định. Ví dụ: *"rất hài lòng nên không có gì không hài lòng"*, *"phòng học thoáng mát , trang thiết bị đầy đủ"* (positive facility mention nhưng bị gán Negative).

2. **Neutral bị nhầm sang cả Negative và Positive** (117 mẫu, 34% lỗi). Neutral chỉ có 167 mẫu test (5.3%), nhưng 70% bị gán sai sang lớp có polarity. Nhiều mẫu Neutral là câu ngắn thiếu sentiment rõ (*"lý thuyết ."*, *"nội dung ."*, *"không ạ ."*) hoặc chứa từ có polarity yếu (*"tâm lý ."*, *"tận tình ."*, *"chi tiết ."*) bị model gán polarity.

3. **Negative ↔ Positive** (163 mẫu gộp cả 2 chiều, 47.6% lỗi). Đây là lỗi nghiêm trọng nhất vì hai lớp đối lập polarity. Phần lớn là Positive → Negative (114) hơn Negative → Positive (49) — gợi ý model thiên vị Negative khi không chắc.

4. **Error vào/vào Neutral** (179 mẫu gộp cả 4 chiều liên quan, 52.3% lỗi). Neutral là lớp yếu nhất (F1 = 0.36).

### 3.3 Representative examples

**Positive → Negative** (lỗi phủ định scope):

| ID | Text | True | Pred |
|---|---|---|---|
| test_00153 | rất hài lòng nên không có gì không hài lòng . | Positive | Negative |
| test_00012 | trong trường macbook thầy số hai thì không có máy nào số một . | Positive | Negative |
| test_00055 | ấn tượng nhất doubledot dạy không cần máy chiếu hay laptop , nhưng lượng bài giảng có thể nói là " khủng " ... | Positive | Negative |
| test_00123 | cần có nhiều giảng viên như cô dạy hơn . | Positive | Negative |
| test_01381 | có giáo trình cho môn học . | Positive | Negative |

Pattern: câu chứa `không`/`chưa` trong cấu trúc phủ định phức tạp, hoặc khen gián tiếp ("cần có nhiều giảng viên như cô" = khen cô nhưng nghe như yêu cầu). TF-IDF chỉ thấy token `không` → lean Negative.

**Neutral → Negative:**

| ID | Text | True | Pred |
|---|---|---|---|
| test_00844 | lý thuyết . | Neutral | Negative |
| test_00938 | cầu kỳ . | Neutral | Negative |
| test_01330 | không ạ . | Neutral | Negative |
| test_02560 | nội dung . | Neutral | Negative |
| test_01612 | giờ là 2014 rồi ! | Neutral | Negative |

**Neutral → Positive:**

| ID | Text | True | Pred |
|---|---|---|---|
| test_00178 | tâm lý . | Neutral | Positive |
| test_01085 | sự tận tình . | Neutral | Positive |
| test_02273 | thầy dạy ổn . | Neutral | Positive |
| test_00931 | nói dễ nghe . | Neutral | Positive |
| test_02603 | nhiều ví dụ . | Neutral | Positive |

Pattern: câu Neutral cực ngắn (1–4 từ), không có sentiment marker rõ. Model gán polarity dựa trên từ đơn lẻ ("tận tình", "ổn", "dễ nghe") có weight Positive trong training. Với Neutral chỉ 5.3% training data, model không học đủ neutral-specific features.

**Negative → Positive:**

| ID | Text | True | Pred |
|---|---|---|---|
| test_02984 | hoạt động nhóm . | Negative | Positive |
| test_01612 | giờ là 2014 rồi ! | Negative | Positive |

Pattern: Negative cũng bị nhầm khi text quá ngắn hoặc chứa từ positive-sounding.

### 3.4 Observed error patterns

| Pattern | Bằng chứng | Số mẫu gần đúng |
|---|---|---|
| **Negation scope** | Câu Positive chứa `không`/`chưa` bị gán Negative: 34/114 P→N errors chứa không/chưa/chẳng | ~34 |
| **Insufficient context (short text)** | Errors ≤5 từ: 56/342 (16.4%), so với 9.0% correct predictions | ~56 |
| **Ambiguous/annotation ambiguity** | Neutral errors chứa từ polarity yếu; cả LR và SVM cùng sai 81% lỗi | ~179 (Neutral-related) |
| **Mixed sentiment** | Mẫu như *"học thì quá ít nhưng khi thi thì quá nhiều"* — chứa cả positive và negative signal | không đếm riêng |

**Length stats (errors vs correct):**

| | Errors | Correct |
|---|---|---|
| Mean words | 13.06 | 14.36 |
| Median words | 11 | 11 |
| ≤5 words | 16.4% | 9.0% |

Errors có xu hướng ngắn hơn correct predictions — 16.4% errors ≤5 từ so với 9.0% correct.

---

## 4. Topic error analysis

### 4.1 Error distribution

- Tổng test: 3,166 mẫu
- Số lỗi: **447** (14.1%)
- Số đúng: 2,719 (85.9%)

### 4.2 Main error pairs

| True → Predicted | Count | % of errors | % of true class |
|---|---|---|---|
| Lecturer → Training_program | 163 | 36.47% | 7.12% |
| Training_program → Lecturer | 101 | 22.60% | 17.66% |
| Others → Training_program | 45 | 10.07% | 28.30% |
| Lecturer → Others | 44 | 9.84% | 1.92% |
| Others → Lecturer | 42 | 9.40% | 26.42% |
| Training_program → Others | 28 | 6.26% | 4.90% |
| Lecturer → Facility | 7 | 1.57% | 0.31% |
| Facility → Training_program | 6 | 1.34% | 4.14% |
| Training_program → Facility | 4 | 0.89% | 0.70% |
| Others → Facility | 3 | 0.67% | 1.89% |
| Facility → Lecturer | 2 | 0.45% | 1.38% |
| Facility → Others | 2 | 0.45% | 1.38% |

**Quan sát chính:**

1. **Lecturer ↔ Training_program chiếm 59% tổng lỗi** (264/447). Hai lớp này overlap nội dung mạnh — cùng nói về giảng dạy, bài tập, slide, thuyết trình. Lecturer tập trung vào người dạy, Training_program tập trung vào chương trình/nội dung học — ranh giới mờ trong nhiều câu ngắn.

2. **Others bị nhầm nặng** (90 mẫu lỗi trên 159 test, 56.6% Others bị sai). Others → Training_program (28.3% Others) và Others → Lecturer (26.4% Others) là hai hướng chính. Others không có từ vựng đặc trưng — nội dung đa dạng (ký túc xá, thời gian, ngoại khóa, thông báo...) bị gán sang lớp có từ overlap.

3. **Facility hầu như không lỗi** (chỉ 10 lỗi Facility-related, 2.2% errors). Facility có từ vựng đặc trưng rõ (phòng, máy chiếu, điều hòa, cơ sở vật chất...) → F1 = 0.92 dù chỉ 145 mẫu test.

4. **Training_program → Lecturer** (101 mẫu) nhiều hơn chiều ngược lại theo tỷ lệ class (17.66% vs 7.12%). Gợi ý Training_program bị "hút" về Lecturer do Lecturer chiếm 72% training data.

### 4.3 Representative examples

**Lecturer → Training_program** (overlap nội dung giảng dạy):

| ID | Text | True | Pred |
|---|---|---|---|
| test_00089 | không slide . | Lecturer | Training_program |
| test_00321 | ít demo . | Lecturer | Training_program |
| test_00520 | có nhiệt huyết . | Lecturer | Training_program |
| test_02591 | dạy bài hay . | Lecturer | Training_program |
| test_02665 | nộp deadline bất chợt . | Lecturer | Training_program |
| test_01908 | bài giảng hấp dẫn . | Lecturer | Training_program |

Pattern: câu ngắn nói về hoạt động giảng dạy mà không đề cập trực tiếp giảng viên → bị gán Training_program. TF-IDF không phân biệt "dạy" là hành động của ai.

**Training_program → Lecturer:**

| ID | Text | True | Pred |
|---|---|---|---|
| test_00570 | slide chi tiết . | Training_program | Lecturer |
| test_00531 | nhiều bài thuyết trình . | Training_program | Lecturer |
| test_00429 | thuyết trình thường xuyên . | Training_program | Lecturer |
| test_02216 | thông báo nghỉ hơi chậm . | Training_program | Lecturer |
| test_03023 | đến lớp đúng giờ ! | Training_program | Lecturer |

Pattern: nội dung mô tả hoạt động lớp học (thuyết trình, slide, điểm danh) mà không rõ là về chương trình hay giảng viên → bị hút về Lecturer (lớp chiếm đa số).

**Others → Training_program / Lecturer:**

| ID | Text | True | Pred |
|---|---|---|---|
| test_00095 | học thì quá ít nhưng khi thi thì quá nhiều yêu cầu viết code... | Others | Training_program |
| test_00169 | các buổi ngoại khóa nên được mở . | Others | Training_program |
| test_00704 | nắng vào buổi sáng ở gần cửa sổ . | Others | Training_program |
| test_00734 | em là sinh viên khó khăn , phải vừa làm vừa học... | Others | Training_program |
| test_00244 | môn này em không được học với cô wzjwz160 . | Others | Training_program |

Pattern: Others rất đa dạng (về mình, về ngoại khóa, về thời tiết, về điều kiện cá nhân) — không có "từ vựng Others" để học. Bất kỳ từ nào overlap với Lecturer/Training_program đều bị hút về lớp lớn.

**Lecturer → Others:**

| ID | Text | True | Pred |
|---|---|---|---|
| test_02254 | hài lòng . | Lecturer | Others |
| test_01085 | sự tận tình . | Lecturer | Others |
| test_01220 | hoàn toàn hài lòng . | Lecturer | Others |
| test_02792 | chấm điểm trên khmtdotuitdotedudotvn . | Lecturer | Others |

Pattern: câu quá ngắn, chỉ có sentiment expression mà không có topic cue → không đủ thông tin để phân loại.

### 4.4 Observed error patterns

| Pattern | Bằng chứng | Số mẫu gần đúng |
|---|---|---|
| **Overlapping topic** | Lecturer ↔ Training_program: 264 mẫu (59%) — cùng domain giảng dạy | ~264 |
| **Others ambiguity** | Others → các lớp khác: 90/159 mẫu bị sai (56.6%) | ~90 |
| **Insufficient context** | Errors ≤5 từ: 65/447 (14.5%), vs 9.0% correct | ~65 |
| **Vague feedback** | Câu chỉ có sentiment, không có topic cue (hài lòng, tận tình...) | ~30 |
| **Majority class bias** | Lecturer chiếm 72% train → model lean về Lecturer khi không chắc | ảnh hưởng chung |

**Length stats (errors vs correct):**

| | Errors | Correct |
|---|---|---|
| Mean words | 14.00 | 14.26 |
| Median words | 11 | 11 |
| ≤5 words | 14.5% | 9.0% |

Tương tự sentiment — errors ngắn hơn correct.

---

## 5. Cross-task observations

### 5.1 Agreement giữa LR và SVM

| Task | SVM errors | LR also wrong | Same wrong prediction | Agreement rate |
|---|---|---|---|---|
| Sentiment | 342 | 300 | 278 | **81.3%** |
| Topic | 447 | 374 | 343 | **76.7%** |

- **81% lỗi sentiment của SVM cũng bị LR sai với cùng predicted label.** Điều này gợi ý phần lớn lỗi không phải do SVM-specific mà là do bản chất mẫu (annotation ambiguity, insufficient context, hoặc feature overlap thực sự).
- Tương tự, **77% lỗi topic** cũng là shared errors.

### 5.2 Common themes

- **Câu ngắn thiếu ngữ cảnh** là pattern chung ở cả hai task — cả sentiment lẫn topic đều khó khi text ≤5 từ.
- **Class imbalance** ảnh hưởng cả hai task: Neutral sentiment (5.3%) và Others topic (5.0%) đều có F1 thấp nhất.
- **Overlap domain** là vấn đề chính: sentiment Neutral bị hút về polarity classes; topic Others bị hút về Lecturer/Training_program.
- **Anonymized tokens** (`wzjwz160`, `doubledot`, `khmtdotuitdotedudotvn`...) xuất hiện trong errors — TF-IDF xử lý chúng như tokens thường, có thể gây noise.

---

## 6. Limitations

1. **Không có confidence score** — prediction files không chứa decision function output, nên không phân tích được lỗi theo confidence. Chỉ biết mẫu sai, không biết model "chắc sai" hay "hầu như đúng".

2. **Không re-annotate** — không thể xác định chính xác mẫu nào là annotation error vs. genuine model error. Tỷ lệ agreement LR-SVM cao (77–81%) gợi ý nhiều mẫu là hard cases nhưng không kết luận được.

3. **TF-IDF không nắm context** — phân tích cho thấy negation scope và multi-aspect feedback là vấn đề, nhưng không quantify được bao nhiêu % lỗi thực sự do limitation này.

4. **Error categories** — phân loại dựa trên quan sát thủ công của representative samples (~30 mẫu mỗi task), không exhaustive. Nhiều mẫu không rõ nguyên nhân → ghi "unclear / insufficient evidence".

5. **Dev set không phân tích** — chỉ phân tích test set. Dev errors có thể có pattern khác.

---

## 7. Implications for next model

Error analysis cho thấy các vấn đề mà Transformer/PhoBERT có thể cần chú ý:

1. **Negation handling** — 34/114 lỗi Positive→Negative chứa `không`/`chưa`. Transformer có thể nắm scope phủ định tốt hơn TF-IDF n-gram qua attention, nhưng đây vẫn là vấn đề khó với câu phủ định phức tạp ("không có gì không hài lòng").

2. **Short-text classification** — 14–16% errors là câu ≤5 từ. Transformer không giải quyết được vấn đề thiếu thông tin, nhưng pre-trained representations có thể giúp hơn TF-IDF cho câu không có từ discriminative.

3. **Class imbalance** — Neutral sentiment (F1 = 0.36) và Others topic (F1 = 0.46) cần xem xét: oversampling, focal loss, hoặc label smoothing có thể phù hợp hơn `class_weight="balanced"`.

4. **Lecturer↔Training_program overlap** — 59% topic errors. Có thể xem xét thêm features hoặc hierarchical classification (phân biệt "về ai" vs "về cái gì").

5. **Others class** — 56.6% mẫu Others bị sai. Others cần định nghĩa rõ hơn hoặc xem xét multi-label approach vì nhiều mẫu Others thực sự mention nhiều aspect.

6. **Anonymized tokens** — `wzjwz...`, `doubledot`, `colonsmile`... có thể cần xử lý đặc biệt hoặc verify tác động.

7. **Annotation ambiguity** — 77–81% lỗi là shared errors giữa hai models khác nhau. Một phần lỗi có thể là annotation noise — Transformer cũng sẽ gặp giới hạn tương tự trên các mẫu này. Cần kỳ vọng thực tế về upper bound performance.

---

## 8. Artifacts

### Files created (`results/error_analysis/`)

| File | Mô tả |
|---|---|
| `sentiment_error_summary.csv` | Error pairs: count, % errors, % true class |
| `topic_error_summary.csv` | Error pairs: count, % errors, % true class |
| `sentiment_error_samples.csv` | 30 representative error samples (top-3 pairs × 10) |
| `topic_error_samples.csv` | 40 representative error samples (top-4 pairs × 10) |
| `error_statistics.json` | Full statistics: pairs, length stats, class counts |

### Source code

| File | Mô tả |
|---|---|
| `src/baseline/error_analysis.py` | Script phân tích — chỉ đọc predictions, không re-train |

### Validation

| Check | Kết quả |
|---|---|
| Sentiment errors = test - correct | ✅ 342 = 3,166 - 2,824 |
| Topic errors = test - correct | ✅ 447 = 3,166 - 2,719 |
| Không duplicate error IDs | ✅ PASS |
| Không sửa test.csv | ✅ UNCHANGED |
| Không sửa prediction files | ✅ READ-ONLY |
| Không sửa data/raw | ✅ 9/9 MD5 UNCHANGED |

---

# ERROR ANALYSIS STATUS: ✅ READY FOR TRANSFORMER
