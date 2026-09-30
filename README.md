# Phân tích cảm xúc và chủ đề từ phản hồi người dùng bằng NLP kết hợp Ontology cảm xúc để hỗ trợ cải tiến dịch vụ

## 1. Project title

Dự án phân tích cảm xúc và chủ đề từ phản hồi người dùng bằng NLP, hướng đến hỗ trợ cải tiến dịch vụ.

Ở trạng thái hiện tại, repository này mới triển khai **MỨC 1**: tiền xử lý, phân tích dữ liệu, mô hình baseline, mô hình Transformer/PhoBERT, đánh giá và phân tích lỗi trên bộ dữ liệu UIT-VSFC. Các thành phần Ontology, KG, RAG hoặc kết hợp luật-ontology **chưa được triển khai** trong mã nguồn hiện tại.

## 2. Objectives

Các mục tiêu đã triển khai:

- **Phân loại cảm xúc**: dự đoán nhãn `Negative`, `Neutral`, `Positive`.
- **Phân loại chủ đề**: dự đoán nhãn `Lecturer`, `Training_program`, `Facility`, `Others`.
- **Tiền xử lý**: chuẩn hóa văn bản, làm sạch và lưu dữ liệu đã xử lý.
- **EDA**: phân tích phân bố nhãn, độ dài văn bản, từ khóa, emoji và quan hệ sentiment-topic.
- **So sánh baseline**: đánh giá mô hình TF-IDF truyền thống.
- **Transformer/PhoBERT**: tinh chỉnh mô hình transformer tiếng Việt cho hai bài toán.
- **Phân tích lỗi**: ghi nhận các dạng lỗi điển hình của mô hình.
- **Hỗ trợ phân tích cải tiến dịch vụ sau này**: cung cấp nền tảng dữ liệu, metric và mô hình cho các bước mở rộng ở MỨC 2.

## 3. Dataset

Dự án sử dụng bộ dữ liệu **UIT-VSFC**.

### Nguồn dữ liệu

Dataset: **UIT-VSFC** (Vietnamese Students' Feedback Corpus)

Nguồn dữ liệu công khai:
https://github.com/kietnv/uit-vsfc

Quy mô sau tiền xử lý:

- Tổng số mẫu: **16.173**
- Train: **11.424**
- Dev: **1.583**
- Test: **3.166**

Nhãn cảm xúc:

- `Negative`
- `Neutral`
- `Positive`

Nhãn chủ đề:

- `Lecturer`
- `Training_program`
- `Facility`
- `Others`

Ghi chú dữ liệu:

- `data/raw/` **không được commit vào Git**.
- Dữ liệu thô cần được lấy từ nguồn công khai UIT-VSFC được trích dẫn trong tài liệu dự án và đặt vào `data/raw/`.
- Tính toàn vẹn dữ liệu thô được kiểm tra bằng **MD5 checksums**.
- Bộ dữ liệu **không có trường thời gian**, do đó hiện chưa thể thực hiện phân tích theo thời gian một cách trực tiếp từ dữ liệu gốc.

## 4. Project pipeline

Pipeline hiện tại trong repository:

```text
Raw UIT-VSFC
→ Preprocessing
→ Validation
→ TF-IDF Baseline
→ Error Analysis
→ EDA
→ PhoBERT Transformer
→ Evaluation
```

Không bao gồm các thành phần chưa triển khai như dashboard, API, ontology-rule fusion, KG hay RAG.

## 5. Repository structure

```text
data/
  processed/

reports/
  DATASET_AUDIT.md
  PREPROCESSING_PLAN.md
  PREPROCESSING_RESULT.md
  BASELINE_RESULT.md
  EDA_RESULT.md
  ERROR_ANALYSIS_RESULT.md
  TRANSFORMER_RESULT.md

results/
  baseline/
  eda/
  error_analysis/
  transformer/

src/
  preprocessing/
  baseline/
  eda/
  transformer/

models/
  baseline/
  transformer/

reference/

requirements.txt
```

Ghi chú:

- `models/` chứa checkpoint/sản phẩm huấn luyện.
- `data/raw/` chứa dữ liệu thô đầu vào.
- `reference/` dùng để lưu tài liệu tham khảo nội bộ.
- `models/`, `data/raw/` và `reference/` **không được Git theo dõi** theo cấu hình `.gitignore`.

## 6. Environment

Môi trường đã kiểm chứng:

- Python **3.12.10**
- Chạy hoàn toàn trên **CPU**
- **Không yêu cầu GPU**
- Huấn luyện PhoBERT trên CPU mất **vài giờ**
- Trên Windows, nếu console gặp lỗi encoding khi in tiếng Việt/emoji, có thể đặt:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

Các phụ thuộc trong `requirements.txt`:

```txt
numpy>=2.0,<3
pandas>=2.2
scikit-learn>=1.5
torch>=2.4
transformers>=5.0
matplotlib>=3.8
joblib>=1.3
regex>=2024.0
```

## 7. Installation

Trên Windows PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Môi trường đã kiểm chứng là **Python 3.12**.

## 8. Dataset preparation

Đặt các file UIT-VSFC thô vào cấu trúc sau:

```text
data/raw/
├── train/
│   ├── sents.txt
│   ├── sentiments.txt
│   └── topics.txt
├── dev/
│   ├── sents.txt
│   ├── sentiments.txt
│   └── topics.txt
└── test/
    ├── sents.txt
    ├── sentiments.txt
    └── topics.txt
```

Dữ liệu thô cần được lấy từ nguồn công khai UIT-VSFC được tham chiếu trong tài liệu dự án. Repository không cung cấp sẵn file dữ liệu thô trong Git.

## 9. How to run

Chạy các bước từ thư mục gốc của project.

### Preprocessing

```bash
python -m src.preprocessing.build_processed
```

### Validation

```bash
python -m src.preprocessing.validate
```

### Baseline

```bash
python -m src.baseline.train_baseline
```

### Error analysis

```bash
python src/baseline/error_analysis.py
```

### EDA

```bash
python src/eda/run_eda.py
```

### EDA validation

```bash
python src/eda/validate.py
```

### Transformer token statistics

```bash
python src/transformer/tokenize_stats.py
```

### Transformer training

```bash
python src/transformer/train_task.py --task sentiment --max-length 32
python src/transformer/train_task.py --task topic --max-length 32
```

Hoặc chạy cả hai task:

```bash
python src/transformer/train_all.py
```

### Transformer evaluation

```bash
python src/transformer/evaluate.py
```

### Transformer validation

```bash
python src/transformer/validate.py
```

Thứ tự pipeline khuyến nghị:

1. `src.preprocessing.build_processed`
2. `src.preprocessing.validate`
3. `src.baseline.train_baseline`
4. `src/baseline/error_analysis.py`
5. `src/eda/run_eda.py`
6. `src/eda/validate.py`
7. `src/transformer/tokenize_stats.py`
8. `src/transformer/train_task.py --task sentiment --max-length 32`
9. `src/transformer/train_task.py --task topic --max-length 32`
10. `src/transformer/evaluate.py`
11. `src/transformer/validate.py`

Lần chạy Transformer đầu tiên sẽ tải `vinai/phobert-base-v2` từ Hugging Face, vì vậy cần **internet một lần**.

## 10. Model configurations

### Baseline

- TF-IDF unigram + bigram
- Logistic Regression
- Linear SVM
- `class_weight=balanced`
- `random_state=42`

### PhoBERT

- `vinai/phobert-base-v2`
- Full fine-tuning
- `max_length=32`
- `batch_size=64`
- `epochs=2`
- `learning_rate=2e-5`
- `weight_decay=0.01`
- `warmup_ratio=0.1`
- Weighted CrossEntropyLoss
- `seed=42`
- Checkpoint được chọn theo **DEV Macro F1**

## 11. Current results

Kết quả test hiện tại:

| Task | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Sentiment | Linear SVM | 0.8920 | 0.7304 | - |
| Sentiment | PhoBERT | 0.9198 | 0.8224 | 0.9226 |
| Topic | Linear SVM | 0.8588 | 0.7531 | - |
| Topic | PhoBERT | 0.8506 | 0.7673 | 0.8603 |

Nhận xét mô tả:

- PhoBERT cải thiện rõ rệt cho bài toán sentiment classification, đặc biệt ở lớp `Neutral`.
- Với bài toán topic classification, PhoBERT có Macro F1 cao hơn Linear SVM nhưng Accuracy thấp hơn một chút.

Tham khảo chi tiết:

- [reports/BASELINE_RESULT.md](reports/BASELINE_RESULT.md)
- [reports/EDA_RESULT.md](reports/EDA_RESULT.md)
- [reports/ERROR_ANALYSIS_RESULT.md](reports/ERROR_ANALYSIS_RESULT.md)
- [reports/TRANSFORMER_RESULT.md](reports/TRANSFORMER_RESULT.md)

## 12. Error analysis

Các phát hiện đã được ghi nhận:

- Lỗi sentiment thường liên quan đến **phạm vi phủ định**, **văn bản ngắn** và điểm yếu của lớp `Neutral`.
- Lỗi topic thường liên quan đến sự **chồng lấn giữa Lecturer và Training_program**, **độ mơ hồ của lớp Others** và **văn bản ngắn**.

## 13. Limitations

Các giới hạn hiện tại:

- UIT-VSFC không có timestamp.
- Dữ liệu tồn tại mất cân bằng lớp.
- PhoBERT được huấn luyện trên CPU.
- Raw dataset không được commit vào Git.
- Mô hình Hugging Face được tham chiếu bằng tên model thay vì revision bất biến.
- Chưa triển khai confidence calibration.
- MỨC 2 ontology/rule fusion chưa được triển khai.

## 14. Future work

Các hướng mở rộng trong tương lai:

- Dashboard
- API/Web demo
- Emotion/Service Feedback Ontology
- Rule-based/Ontology fusion
- Explainability
- Robustness/Ablation

Các hạng mục trên là **future work**, chưa phải tính năng đã triển khai trong repository hiện tại.

## 15. Reproducibility

Các yếu tố hỗ trợ tái lập:

- Random seed cố định `42`
- Kiểm tra MD5 cho dữ liệu raw
- `data/processed/` được Git theo dõi
- Các model sinh ra được loại khỏi Git
- `results/` và `reports/` được Git theo dõi

## 16. License / academic note

Repository này là một **academic project**. Bộ dữ liệu vẫn chịu các điều khoản nguồn, license và điều kiện sử dụng gốc của nó.
