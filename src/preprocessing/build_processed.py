"""
build_processed.py — Pipeline chính: load → clean → export

Đề tài #2: Phân tích cảm xúc và chủ đề từ phản hồi người dùng
Theo PREPROCESSING_PLAN.md đã duyệt 2026-09-29

Chạy: python -m src.preprocessing.build_processed
Hoặc: python src/preprocessing/build_processed.py
"""

import os
import sys
import json
import csv
from datetime import datetime, timezone

import pandas as pd

# Thêm project root vào path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.load_raw import load_all_splits, verify_raw_checksums
from src.preprocessing.text_cleaner import clean_text, to_lowercase


# === Đường dẫn ===
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# === Annotation conflict cần xóa (Q1=A đã duyệt) ===
# Train index 11293 (sentiment=2, topic=0) và index 11417 (sentiment=0, topic=0)
CONFLICT_INDICES_TRAIN = [11293, 11417]
CONFLICT_TEXT = (
    "thầy dạy hay , tuy nhiên còn nhiều chỗ chưa thật sự giải đáp "
    "hoàn toàn cho sinh viên vì chưa đủ thời gian ."
)

# === Label maps ===
LABEL_MAPS = {
    "sentiment": {"0": "Negative", "1": "Neutral", "2": "Positive"},
    "topic": {
        "0": "Lecturer",
        "1": "Training_program",
        "2": "Facility",
        "3": "Others",
    },
}


def remove_annotation_conflicts(df_train: pd.DataFrame) -> tuple:
    """
    Xóa 2 dòng annotation conflict trong train (Q1=A đã duyệt).
    KHÔNG sửa data/raw/ — chỉ xóa trong DataFrame.

    Returns:
        (df_cleaned, removed_records)
    """
    removed_records = []

    for idx in CONFLICT_INDICES_TRAIN:
        row = df_train[df_train["id"] == f"train_{idx:05d}"]
        if len(row) == 0:
            raise RuntimeError(
                f"Conflict row train_{idx:05d} not found! "
                f"DỪNG — không thể xử lý theo plan."
            )

        row_data = row.iloc[0]

        # Xác nhận đúng text conflict
        if row_data["text"] != CONFLICT_TEXT:
            raise RuntimeError(
                f"Conflict row train_{idx:05d} text mismatch!\n"
                f"Expected: {CONFLICT_TEXT!r}\n"
                f"Got: {row_data['text']!r}\n"
                f"DỪNG — không thể xử lý theo plan."
            )

        removed_records.append({
            "split": "train",
            "id": row_data["id"],
            "original_index": idx,
            "text": row_data["text"],
            "sentiment": int(row_data["sentiment"]),
            "topic": int(row_data["topic"]),
        })

    # Xóa các dòng conflict
    conflict_ids = [f"train_{idx:05d}" for idx in CONFLICT_INDICES_TRAIN]
    df_cleaned = df_train[~df_train["id"].isin(conflict_ids)].copy()
    df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned, removed_records


def apply_text_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng text preprocessing pipeline:
    - text_clean: NFC + emoji→Unicode + whitespace (giữ case)
    - text_lower: lowercase (cho TF-IDF)
    """
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_lower"] = df["text_clean"].apply(to_lowercase)
    return df


def compute_class_weights(df_train: pd.DataFrame) -> dict:
    """
    Tính class weights theo công thức: weight = total / (num_classes * count).
    Tương đương sklearn compute_class_weight('balanced').
    """
    n = len(df_train)

    # Sentiment weights
    sent_counts = df_train["sentiment"].value_counts().to_dict()
    n_sent_classes = 3
    sentiment_weights = {}
    for cls in [0, 1, 2]:
        count = sent_counts.get(cls, 0)
        weight = n / (n_sent_classes * count) if count > 0 else 0.0
        sentiment_weights[str(cls)] = round(weight, 4)

    # Topic weights
    topic_counts = df_train["topic"].value_counts().to_dict()
    n_topic_classes = 4
    topic_weights = {}
    for cls in [0, 1, 2, 3]:
        count = topic_counts.get(cls, 0)
        weight = n / (n_topic_classes * count) if count > 0 else 0.0
        topic_weights[str(cls)] = round(weight, 4)

    return {
        "sentiment_weights": sentiment_weights,
        "topic_weights": topic_weights,
        "method": "inverse_frequency",
        "formula": "weight = total_samples / (num_classes * count_per_class)",
        "train_total": n,
        "note": "Chỉ tính trên train set. Áp dụng khi train model, KHÔNG sửa data.",
    }


def compute_stats(splits: dict) -> dict:
    """Tính thống kê tổng hợp sau xử lý."""
    stats = {
        "total_samples": {},
        "sentiment_distribution": {},
        "topic_distribution": {},
        "text_length_stats": {},
    }

    total = 0
    for split_name, df in splits.items():
        n = len(df)
        total += n
        stats["total_samples"][split_name] = n

        # Sentiment distribution
        sent_dist = df["sentiment"].value_counts().sort_index().to_dict()
        stats["sentiment_distribution"][split_name] = {
            str(k): v for k, v in sent_dist.items()
        }

        # Topic distribution
        topic_dist = df["topic"].value_counts().sort_index().to_dict()
        stats["topic_distribution"][split_name] = {
            str(k): v for k, v in topic_dist.items()
        }

        # Text length stats (on text_clean)
        lengths = df["text_clean"].str.len()
        stats["text_length_stats"][split_name] = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 1),
            "median": round(float(lengths.median()), 1),
        }

    stats["total_samples"]["total"] = total
    return stats


def count_emoji_affected(splits_before: dict, splits_after: dict) -> dict:
    """Đếm số câu có text thay đổi do emoji replacement."""
    result = {}
    for split_name in splits_before:
        df_before = splits_before[split_name]
        df_after = splits_after[split_name]
        # So sánh text vs text_clean (sau khi cả NFC + emoji + whitespace)
        changed = (df_after["text"] != df_after["text_clean"]).sum()
        result[split_name] = int(changed)
    return result


def export_csv(df: pd.DataFrame, filepath: str) -> None:
    """Xuất DataFrame ra CSV theo format đã duyệt."""
    # Chọn đúng cột và thứ tự theo schema
    columns = ["id", "text", "text_clean", "text_lower", "sentiment", "topic"]
    df_out = df[columns]

    df_out.to_csv(
        filepath,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONNUMERIC,
        lineterminator="\n",
    )


def build_preprocessing_log(
    samples_before: dict,
    samples_after: dict,
    removed_records: list,
    emoji_affected: dict,
    created_at: str,
) -> dict:
    """Tạo preprocessing log chi tiết."""
    return {
        "created_at": created_at,
        "source": "data/raw/",
        "output": "data/processed/",
        "decisions": {
            "Q1_annotation_conflict": "A - Xóa cả 2 dòng",
            "Q2_emoji_acronym": "A - Chuyển về Emoji Unicode",
            "Q3_punctuation_spacing": "A - Giữ nguyên pre-tokenized",
            "Q4_lowercase": "C - Tạo 2 phiên bản (text_clean + text_lower)",
        },
        "steps": [
            {
                "step": 1,
                "name": "load_and_merge",
                "description": (
                    "Đọc sents.txt + sentiments.txt + topics.txt, "
                    "ghép theo index dòng"
                ),
                "samples": samples_before,
            },
            {
                "step": 2,
                "name": "verify_raw_checksums",
                "description": "Kiểm tra MD5 checksum 9 file raw — tất cả khớp",
            },
            {
                "step": 3,
                "name": "remove_annotation_conflict",
                "description": (
                    "Xóa 2 dòng conflict trong train "
                    "(index 11293, 11417) — Q1=A đã duyệt"
                ),
                "removed": removed_records,
                "reason": (
                    "Annotation conflict: cùng câu, "
                    "nhãn sentiment mâu thuẫn (positive vs negative)"
                ),
                "samples_after": {
                    "train": samples_after["train"],
                    "dev": samples_after["dev"],
                    "test": samples_after["test"],
                },
            },
            {
                "step": 4,
                "name": "unicode_normalization",
                "description": "NFC normalization cho tất cả câu",
            },
            {
                "step": 5,
                "name": "emoji_acronym_replacement",
                "description": (
                    "Chuyển acronym về Emoji Unicode — Q2=A đã duyệt"
                ),
                "affected_samples": emoji_affected,
            },
            {
                "step": 6,
                "name": "whitespace_normalization",
                "description": "Chuẩn hóa khoảng trắng: strip + collapse multiple spaces",
            },
            {
                "step": 7,
                "name": "punctuation_spacing",
                "description": (
                    "Giữ nguyên format pre-tokenized — Q3=A đã duyệt. "
                    "Không thay đổi."
                ),
            },
            {
                "step": 8,
                "name": "generate_lowercase",
                "description": (
                    "Tạo cột text_lower = text_clean.lower() — Q4=C đã duyệt"
                ),
            },
        ],
    }


def main():
    """Pipeline chính."""
    print("=" * 60)
    print("PREPROCESSING PIPELINE — UIT-VSFC")
    print("Đề tài #2 — Theo PREPROCESSING_PLAN.md đã duyệt")
    print("=" * 60)

    created_at = datetime.now(timezone.utc).isoformat()

    # === Bước 0: Tạo thư mục output ===
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"\n[0] Output dir: {PROCESSED_DIR}")

    # === Bước 1: Kiểm tra raw checksums + Load data ===
    print("\n[1] Verifying raw data checksums...")
    verify_raw_checksums(RAW_DIR)
    print("    ✅ All 9 raw files match expected checksums")

    print("\n[2] Loading raw data...")
    splits = load_all_splits(RAW_DIR)
    samples_before = {name: len(df) for name, df in splits.items()}
    for name, df in splits.items():
        print(f"    {name}: {len(df)} samples loaded")

    # === Bước 2: Xóa annotation conflict (Q1=A) ===
    print("\n[3] Removing annotation conflicts (Q1=A)...")
    splits["train"], removed_records = remove_annotation_conflicts(splits["train"])
    print(f"    Removed {len(removed_records)} rows from train:")
    for rec in removed_records:
        print(f"      - {rec['id']}: sentiment={rec['sentiment']}, topic={rec['topic']}")

    samples_after_conflict = {name: len(df) for name, df in splits.items()}
    print(f"    Train: {samples_before['train']} → {samples_after_conflict['train']}")

    # === Bước 3: Text preprocessing ===
    print("\n[4] Applying text preprocessing...")
    print("    - Unicode NFC normalization")
    print("    - Emoji acronym → Emoji Unicode (Q2=A)")
    print("    - Whitespace normalization")
    print("    - Punctuation spacing: giữ nguyên (Q3=A)")

    # Lưu bản trước preprocessing để đếm affected
    splits_before_clean = {name: df.copy() for name, df in splits.items()}

    for name in splits:
        splits[name] = apply_text_preprocessing(splits[name])
        changed = (splits[name]["text"] != splits[name]["text_clean"]).sum()
        print(f"    {name}: {changed} sentences changed by text cleaning")

    emoji_affected = count_emoji_affected(splits_before_clean, splits)

    # === Bước 4: Tạo text_lower (Q4=C) ===
    print("\n[5] text_lower already generated in preprocessing step (Q4=C)")

    # === Bước 5: Export CSVs ===
    print("\n[6] Exporting CSV files...")
    for name, df in splits.items():
        filepath = os.path.join(PROCESSED_DIR, f"{name}.csv")
        export_csv(df, filepath)
        print(f"    ✅ {filepath} ({len(df)} rows)")

    # === Bước 6: Export label_maps.json ===
    print("\n[7] Exporting label_maps.json...")
    label_maps_path = os.path.join(PROCESSED_DIR, "label_maps.json")
    with open(label_maps_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAPS, f, ensure_ascii=False, indent=2)
    print(f"    ✅ {label_maps_path}")

    # === Bước 7: Export class_weights.json ===
    print("\n[8] Computing and exporting class_weights.json...")
    class_weights = compute_class_weights(splits["train"])
    class_weights_path = os.path.join(PROCESSED_DIR, "class_weights.json")
    with open(class_weights_path, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, ensure_ascii=False, indent=2)
    print(f"    ✅ {class_weights_path}")
    print(f"    Sentiment weights: {class_weights['sentiment_weights']}")
    print(f"    Topic weights: {class_weights['topic_weights']}")

    # === Bước 8: Export stats_summary.json ===
    print("\n[9] Computing and exporting stats_summary.json...")
    stats = compute_stats(splits)
    stats["removed_conflicts"] = len(removed_records)
    stats["emoji_acronyms_affected"] = emoji_affected
    stats_path = os.path.join(PROCESSED_DIR, "stats_summary.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"    ✅ {stats_path}")

    # === Bước 9: Export preprocessing_log.json ===
    print("\n[10] Exporting preprocessing_log.json...")
    samples_after = {name: len(df) for name, df in splits.items()}
    log = build_preprocessing_log(
        samples_before=samples_before,
        samples_after=samples_after,
        removed_records=removed_records,
        emoji_affected=emoji_affected,
        created_at=created_at,
    )
    log_path = os.path.join(PROCESSED_DIR, "preprocessing_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"    ✅ {log_path}")

    # === Bước 10: Kiểm tra lại raw checksums ===
    print("\n[11] Final raw data integrity check...")
    verify_raw_checksums(RAW_DIR)
    print("    ✅ All 9 raw files STILL match expected checksums")
    print("    ✅ data/raw/ was NOT modified")

    # === Hoàn tất ===
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"  Train: {samples_after['train']} samples")
    print(f"  Dev:   {samples_after['dev']} samples")
    print(f"  Test:  {samples_after['test']} samples")
    print(f"  Total: {sum(samples_after.values())} samples")
    print(f"  Output: {PROCESSED_DIR}")
    print("=" * 60)
    print("\n→ Chạy validate.py để kiểm tra kết quả.")


if __name__ == "__main__":
    main()
