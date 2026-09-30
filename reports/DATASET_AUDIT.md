# DATASET AUDIT

**Đề tài #2:** Phân tích cảm xúc và chủ đề từ phản hồi người dùng bằng NLP kết hợp Ontology cảm xúc để hỗ trợ cải tiến dịch vụ

**Dataset:** UIT-VSFC (Vietnamese Students' Feedback Corpus)

**Ngày kiểm tra:** 2026-09-29

---

## 1. Dataset structure

```
data/raw/
├── README.txt                  # Mô tả cấu trúc và nhãn
├── train/
│   ├── sents.txt               # 898,090 bytes
│   ├── sentiments.txt          #  22,852 bytes
│   └── topics.txt              #  22,852 bytes
├── dev/
│   ├── sents.txt               # 118,628 bytes
│   ├── sentiments.txt          #   3,166 bytes
│   └── topics.txt              #   3,166 bytes
└── test/
    ├── sents.txt               # 247,849 bytes
    ├── sentiments.txt          #   6,332 bytes
    └── topics.txt              #   6,332 bytes
```

**Định dạng:** Plain text, mỗi file chứa một giá trị trên mỗi dòng. Ba file trong mỗi split được căn chỉnh theo số dòng (dòng thứ *i* trong `sents.txt` tương ứng với dòng thứ *i* trong `sentiments.txt` và `topics.txt`).

**Encoding:**
- `sents.txt`: UTF-8 (không BOM)
- `sentiments.txt`: US-ASCII (chỉ chứa ký tự 0/1/2)
- `topics.txt`: US-ASCII (chỉ chứa ký tự 0/1/2/3)

**Line endings:** Tất cả file sử dụng LF (`\n`), không có CRLF. Tất cả kết thúc bằng newline.

**Các field dữ liệu (3 file riêng biệt):**

| File | Nội dung | Vai trò |
|---|---|---|
| `sents.txt` | Câu phản hồi gốc (tiếng Việt, đã tokenize) | Nội dung phản hồi |
| `sentiments.txt` | Nhãn cảm xúc (0, 1, 2) | Nhãn sentiment |
| `topics.txt` | Nhãn chủ đề (0, 1, 2, 3) | Nhãn topic |

---

## 2. Train

| Thông số | Giá trị |
|---|---|
| Số mẫu | **11,426** |
| Tỷ lệ trong tổng | 70.6% |
| Dòng trống | 0 |
| Nhãn sentiment không hợp lệ | 0 |
| Nhãn topic không hợp lệ | 0 |
| Câu trùng lặp (trong split) | **1** (2 dòng cùng nội dung, **khác nhãn sentiment** — xem mục 7) |
| Độ dài câu (ký tự) | min=4, max=660, trung bình=59.1 |
| Câu chứa emoji acronym | 148 |

**Phân bố sentiment (train):**
| Nhãn | Giá trị | Số mẫu | Tỷ lệ |
|---|---|---|---|
| Negative | 0 | 5,325 | 46.6% |
| Neutral | 1 | 458 | 4.0% |
| Positive | 2 | 5,643 | 49.4% |

**Phân bố topic (train):**
| Nhãn | Giá trị | Số mẫu | Tỷ lệ |
|---|---|---|---|
| Lecturer | 0 | 8,166 | 71.5% |
| Training program | 1 | 2,201 | 19.3% |
| Facility | 2 | 497 | 4.3% |
| Others | 3 | 562 | 4.9% |

---

## 3. Dev

| Thông số | Giá trị |
|---|---|
| Số mẫu | **1,583** |
| Tỷ lệ trong tổng | 9.8% |
| Dòng trống | 0 |
| Nhãn không hợp lệ | 0 |
| Câu trùng lặp (trong split) | 0 |
| Độ dài câu (ký tự) | min=4, max=718, trung bình=56.4 |
| Câu chứa emoji acronym | 11 |

**Phân bố sentiment (dev):**
| Nhãn | Giá trị | Số mẫu | Tỷ lệ |
|---|---|---|---|
| Negative | 0 | 705 | 44.5% |
| Neutral | 1 | 73 | 4.6% |
| Positive | 2 | 805 | 50.9% |

**Phân bố topic (dev):**
| Nhãn | Giá trị | Số mẫu | Tỷ lệ |
|---|---|---|---|
| Lecturer | 0 | 1,151 | 72.7% |
| Training program | 1 | 267 | 16.9% |
| Facility | 2 | 70 | 4.4% |
| Others | 3 | 95 | 6.0% |

---

## 4. Test

| Thông số | Giá trị |
|---|---|
| Số mẫu | **3,166** |
| Tỷ lệ trong tổng | 19.6% |
| Dòng trống | 0 |
| Nhãn không hợp lệ | 0 |
| Câu trùng lặp (trong split) | 0 |
| Độ dài câu (ký tự) | min=4, max=411, trung bình=58.8 |
| Câu chứa emoji acronym | 51 |

**Phân bố sentiment (test):**
| Nhãn | Giá trị | Số mẫu | Tỷ lệ |
|---|---|---|---|
| Negative | 0 | 1,409 | 44.5% |
| Neutral | 1 | 167 | 5.3% |
| Positive | 2 | 1,590 | 50.2% |

**Phân bố topic (test):**
| Nhãn | Giá trị | Số mẫu | Tỷ lệ |
|---|---|---|---|
| Lecturer | 0 | 2,290 | 72.3% |
| Training program | 1 | 572 | 18.1% |
| Facility | 2 | 145 | 4.6% |
| Others | 3 | 159 | 5.0% |

---

## 5. Sentiment labels

| Giá trị | Ý nghĩa | Tổng (train+dev+test) | Tỷ lệ tổng |
|---|---|---|---|
| 0 | Negative (tiêu cực) | 7,439 | 46.0% |
| 1 | Neutral (trung lập) | 698 | 4.3% |
| 2 | Positive (tích cực) | 8,038 | 49.7% |

**Số lượng lớp sentiment: 3** (negative, neutral, positive)

**Nhận xét:**
- Lớp **Neutral cực kỳ mất cân bằng** — chỉ chiếm ~4% tổng số mẫu (698/16,175).
- Negative và Positive gần cân bằng (~46% vs ~50%).
- Phân bố nhất quán giữa train/dev/test (stratified split).

---

## 6. Topic labels

| Giá trị | Ý nghĩa | Tổng (train+dev+test) | Tỷ lệ tổng |
|---|---|---|---|
| 0 | Lecturer (giảng viên) | 11,607 | 71.8% |
| 1 | Training program (chương trình đào tạo) | 3,040 | 18.8% |
| 2 | Facility (cơ sở vật chất) | 712 | 4.4% |
| 3 | Others (khác) | 816 | 5.0% |

**Số lượng lớp topic: 4** (lecturer, training program, facility, others)

**Nhận xét:**
- Lớp **Lecturer chiếm áp đảo** (~72%), gây mất cân bằng nghiêm trọng.
- Facility và Others có rất ít mẫu (~4-5% mỗi lớp).
- Phân bố nhất quán giữa train/dev/test (stratified split).

---

## 7. Data quality

### 7.1 Dữ liệu thiếu / null
- **Không có dòng trống** trong bất kỳ file nào.
- **Không có giá trị null** hay missing.
- Tất cả 3 file trong mỗi split đều có **cùng số dòng** (alignment đúng).

### 7.2 Dòng trùng lặp
- **Train:** 1 cặp trùng lặp (dòng 11,294 và 11,418).
  - Nội dung: *"thầy dạy hay , tuy nhiên còn nhiều chỗ chưa thật sự giải đáp hoàn toàn cho sinh viên vì chưa đủ thời gian ."*
  - ⚠️ **Conflict nhãn:** Dòng 11,294 gán sentiment=**2** (positive), dòng 11,418 gán sentiment=**0** (negative). Cùng topic=0 (lecturer).
  - Đây là **annotation conflict** — cùng một câu nhưng hai annotator gán nhãn khác nhau.
- **Dev:** Không có trùng lặp.
- **Test:** Không có trùng lặp.

### 7.3 Encoding
- Tất cả sents.txt: UTF-8 (không BOM) — phù hợp cho tiếng Việt.
- Tất cả sentiments.txt / topics.txt: US-ASCII.
- Line endings: LF (`\n`) nhất quán trên toàn bộ 9 file.

### 7.4 Dấu hiệu bất thường
- **Emoji đã được thay thế bằng acronym** (theo README.txt): `colonsmile`, `colonsad`, `colonlove`, `dotdotdot`, `doubledot`, v.v. — tổng 210 câu chứa acronym (148 train, 11 dev, 51 test).
- **Câu rất ngắn:** Có một số câu chỉ 4-5 ký tự (ví dụ: "dễ .", "tệ .", "hay .", "có .") — đây là phản hồi ngắn thực tế, không phải lỗi.
- **Một số câu tiếng Anh:** Khoảng 21 câu (14 train, 4 dev, 3 test) có vẻ chỉ chứa ký tự ASCII, có thể là phản hồi bằng tiếng Anh (ví dụ: "good .", "funny ."). Đây là đặc điểm tự nhiên của dữ liệu sinh viên.
- **Văn bản đã tokenize:** Dữ liệu có khoảng trắng trước dấu chấm câu (ví dụ: "dễ .") — đã qua bước tokenize sẵn.
- **Typo "cshrap":** Trong README.txt, acronym cho `c#` ghi là "cshrap" (có thể là lỗi chính tả, đúng ra là "csharp"). Đây là lỗi trong README, không ảnh hưởng dữ liệu.

---

## 8. Data leakage check

| Kiểm tra | Kết quả |
|---|---|
| Train ∩ Dev (câu trùng) | **0** — Không có leakage |
| Train ∩ Test (câu trùng) | **0** — Không có leakage |
| Dev ∩ Test (câu trùng) | **0** — Không có leakage |

**Kết luận: KHÔNG CÓ DATA LEAKAGE giữa các split.**

Phân bố nhãn giữa train/dev/test nhất quán (chênh lệch <3% giữa các split), cho thấy dữ liệu được chia theo **stratified split** — đây là thực hành tốt.

**Tỷ lệ split:** ~70/10/20 (train/dev/test) — hợp lý cho bài toán NLP.

---

## 9. Đối chiếu yêu cầu Mức 1

Yêu cầu Mức 1 từ đề cương:

| # | Yêu cầu đề cương | Thực tế dataset | Đáp ứng |
|---|---|---|---|
| 1 | "Thu thập hoặc sử dụng corpus phản hồi tiếng Việt/tiếng Anh" | UIT-VSFC — corpus phản hồi sinh viên tiếng Việt, 16,175 mẫu | ✅ Đáp ứng |
| 2 | "Chuẩn hóa nhãn sentiment và topic" | Sentiment: 3 lớp (0,1,2). Topic: 4 lớp (0,1,2,3). Đã có nhãn sẵn | ✅ Đáp ứng |
| 3 | "Tiền xử lý, EDA" | Dữ liệu thô đã có, đủ để thực hiện EDA | ✅ Sẵn sàng |
| 4 | "Xây baseline TF-IDF + Logistic Regression/SVM" | Dataset đủ lớn (11k train) để xây baseline | ✅ Sẵn sàng |
| 5 | "Thử một mô hình Transformer (PhoBERT/BERT/DistilBERT)" | Dữ liệu tiếng Việt, phù hợp với PhoBERT | ✅ Sẵn sàng |
| 6 | "Dashboard theo thời gian, chủ đề, mức sentiment" | Dataset không có trường thời gian | ⚠️ Thiếu chiều thời gian |
| 7 | "Tách riêng đánh giá sentiment và topic" | Hai file nhãn riêng biệt | ✅ Đáp ứng |
| 8 | "Macro-F1/weighted-F1, confusion matrix" | Có train/dev/test split sẵn, metric tính được | ✅ Sẵn sàng |
| 9 | "Chuẩn hóa Unicode, xử lý emoji/teencode" | Emoji đã chuyển thành acronym; cần xử lý tiếp | ⚠️ Cần tiền xử lý |
| 10 | "Chống data leakage; lưu random seed" | Không có leakage giữa split | ✅ Đáp ứng |

**Ghi chú quan trọng:**
- Đề cương gợi ý dùng UIT-VSFC ([1], [2]) — dataset hiện tại **khớp đúng** với nguồn được đề cương chỉ định.
- Dataset thuộc miền **phản hồi sinh viên về giảng dạy** — phù hợp với mục tiêu "phân tích phản hồi người dùng để cải tiến dịch vụ" (dịch vụ = giáo dục).

---

## 10. Vấn đề cần xử lý

### 🔴 Mức cao (phải xử lý trước khi train)

| # | Vấn đề | Chi tiết | Đề xuất |
|---|---|---|---|
| 1 | **Annotation conflict** | Train dòng 11,294 và 11,418: cùng câu, nhãn sentiment mâu thuẫn (positive vs negative) | Cần quyết định giữ/xóa dòng nào hoặc xóa cả hai. Ghi lại quyết định trong preprocessing log |
| 2 | **Class imbalance — Neutral** | Sentiment neutral chỉ ~4% (698 mẫu). Mô hình sẽ khó học lớp này | Cần dùng weighted loss, oversampling, hoặc báo cáo per-class metric. Macro-F1 (yêu cầu đề cương) sẽ phản ánh đúng |
| 3 | **Class imbalance — Topic** | Lecturer chiếm ~72%, Facility chỉ ~4.4% (712 mẫu) | Tương tự sentiment: cần class weighting hoặc stratified sampling |

### 🟡 Mức trung bình (nên xử lý)

| # | Vấn đề | Chi tiết | Đề xuất |
|---|---|---|---|
| 4 | **Emoji acronym** | 210 câu chứa acronym như `colonsmile`, `dotdotdot` — không phải từ tiếng Việt tự nhiên | Có thể: (a) giữ nguyên nếu model học được, (b) chuyển về emoji gốc, (c) chuyển về text mô tả |
| 5 | **Văn bản đã pre-tokenize** | Khoảng trắng trước dấu chấm câu; cần cân nhắc khi dùng với PhoBERT tokenizer | Kiểm tra xem PhoBERT tokenizer có xử lý tốt format này không |
| 6 | **Thiếu chiều thời gian** | Đề cương yêu cầu "dashboard theo thời gian" nhưng dataset không có timestamp | Dashboard sẽ thiếu trục thời gian; cần ghi nhận giới hạn này trong báo cáo |

### 🟢 Mức thấp (ghi nhận)

| # | Vấn đề | Chi tiết |
|---|---|---|
| 7 | Câu rất ngắn | Một số câu chỉ 4-5 ký tự ("dễ .", "tệ .") — là dữ liệu thực, không phải lỗi |
| 8 | Câu tiếng Anh | ~21 câu có thể là tiếng Anh — dữ liệu thực, số lượng không đáng kể |
| 9 | Typo trong README | "cshrap" thay vì "csharp" — chỉ ảnh hưởng README, không ảnh hưởng dữ liệu |

---

## 11. Kết luận

### Tổng quan

| Thông số | Giá trị |
|---|---|
| Tên dataset | UIT-VSFC (Vietnamese Students' Feedback Corpus) |
| Ngôn ngữ | Tiếng Việt (có một số ít câu tiếng Anh) |
| Miền | Phản hồi sinh viên về giảng dạy đại học |
| Tổng số mẫu | **16,175** |
| Train / Dev / Test | 11,426 / 1,583 / 3,166 (70.6% / 9.8% / 19.6%) |
| Số lớp sentiment | **3** (negative, neutral, positive) |
| Số lớp topic | **4** (lecturer, training program, facility, others) |
| Data leakage | **Không có** |
| Missing values | **Không có** |
| Encoding | UTF-8 / ASCII, LF, không BOM |

### Đánh giá tổng thể

**Dataset phù hợp** để thực hiện Mức 1 của đề tài #2. Đây chính là dataset UIT-VSFC được đề cương chỉ định. Dữ liệu sạch, có cấu trúc rõ ràng, có cả nhãn sentiment lẫn topic, split train/dev/test đã sẵn sàng và không có data leakage.

**Hai vấn đề chính** cần giải quyết trước khi tiến hành training:
1. Xử lý 1 cặp câu trùng lặp có nhãn mâu thuẫn trong tập train.
2. Lên chiến lược đối phó class imbalance (đặc biệt neutral sentiment và các lớp topic thiểu số).

**Không được** chỉnh sửa dữ liệu gốc trong `data/raw/`. Mọi xử lý phải được thực hiện trên bản sao trong pipeline tiền xử lý.
