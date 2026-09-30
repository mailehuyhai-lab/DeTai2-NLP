"""
validate.py — Kiểm tra output preprocessing vs raw (integrity + quality checks)

Đề tài #2: Phân tích cảm xúc và chủ đề từ phản hồi người dùng
Theo PREPROCESSING_PLAN.md đã duyệt 2026-09-29

Chạy: python -m src.preprocessing.validate
Hoặc: python src/preprocessing/validate.py

11 Integrity checks (PHẢI PASS) + 5 Quality checks (cảnh báo)
"""

import os
import sys
import csv
import json

import pandas as pd

# Thêm project root vào path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.load_raw import verify_raw_checksums, RAW_CHECKSUMS
from src.preprocessing.text_cleaner import ALL_ACRONYM_KEYS

# === Đường dẫn ===
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# === Số mẫu kỳ vọng (theo plan) ===
EXPECTED_COUNTS = {
    "train": 11424,
    "dev": 1583,
    "test": 3166,
}

# === Phân bố nhãn kỳ vọng dev/test (phải giữ nguyên từ raw) ===
EXPECTED_DEV_SENTIMENT = {"0": 705, "1": 73, "2": 805}
EXPECTED_DEV_TOPIC = {"0": 1151, "1": 267, "2": 70, "3": 95}
EXPECTED_TEST_SENTIMENT = {"0": 1409, "1": 167, "2": 1590}
EXPECTED_TEST_TOPIC = {"0": 2290, "1": 572, "2": 145, "3": 159}

# === Conflict text (phải không tồn tại trong processed train) ===
CONFLICT_TEXT = (
    "thầy dạy hay , tuy nhiên còn nhiều chỗ chưa thật sự giải đáp "
    "hoàn toàn cho sinh viên vì chưa đủ thời gian ."
)


class ValidationResult:
    """Kết quả một check."""

    def __init__(self, check_id: str, name: str):
        self.check_id = check_id
        self.name = name
        self.passed = False
        self.message = ""

    def pass_(self, msg: str = ""):
        self.passed = True
        self.message = msg

    def fail_(self, msg: str = ""):
        self.passed = False
        self.message = msg

    def __str__(self):
        status = "[PASS]" if self.passed else "[FAIL]"
        line = f"{status} {self.check_id}: {self.name}"
        if self.message:
            line += f" — {self.message}"
        return line


def load_processed_csv(split: str) -> pd.DataFrame:
    """Đọc file CSV processed."""
    filepath = os.path.join(PROCESSED_DIR, f"{split}.csv")
    df = pd.read_csv(filepath, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    # QUOTE_NONNUMERIC causes int columns to be read as float → convert back
    df["sentiment"] = df["sentiment"].astype(int)
    df["topic"] = df["topic"].astype(int)
    return df


def run_integrity_checks() -> list:
    """11 Integrity checks — PHẢI PASS."""
    results = []

    # I-1: Raw không bị sửa
    r = ValidationResult("I-1", "Raw files unchanged")
    try:
        checksums = verify_raw_checksums(RAW_DIR)
        matched = sum(1 for v in checksums.values() if v["match"])
        r.pass_(f"{matched}/{len(checksums)} checksums match")
    except Exception as e:
        r.fail_(str(e))
    results.append(r)

    # Load processed data
    try:
        splits = {s: load_processed_csv(s) for s in ["train", "dev", "test"]}
    except Exception as e:
        r2 = ValidationResult("I-2", "Load processed CSVs")
        r2.fail_(f"Cannot load CSVs: {e}")
        results.append(r2)
        return results  # Cannot continue

    # I-2: Số mẫu đúng
    r = ValidationResult("I-2", "Sample counts correct")
    counts = {s: len(df) for s, df in splits.items()}
    if counts == EXPECTED_COUNTS:
        r.pass_(
            f"train={counts['train']}, dev={counts['dev']}, test={counts['test']}"
        )
    else:
        r.fail_(f"Expected {EXPECTED_COUNTS}, got {counts}")
    results.append(r)

    # I-3: Không có null/NaN
    r = ValidationResult("I-3", "No null/NaN values")
    total_nulls = sum(df.isnull().sum().sum() for df in splits.values())
    if total_nulls == 0:
        r.pass_("0 null values across all CSVs")
    else:
        r.fail_(f"{total_nulls} null values found")
    results.append(r)

    # I-4: Không có dòng trống
    r = ValidationResult("I-4", "No empty text strings")
    empty_count = 0
    for s, df in splits.items():
        for col in ["text", "text_clean", "text_lower"]:
            empties = (df[col].astype(str).str.strip() == "").sum()
            empty_count += empties
    if empty_count == 0:
        r.pass_("0 empty strings in text columns")
    else:
        r.fail_(f"{empty_count} empty strings found")
    results.append(r)

    # I-5: Nhãn hợp lệ
    r = ValidationResult("I-5", "Labels valid")
    invalid = 0
    for s, df in splits.items():
        bad_sent = (~df["sentiment"].isin([0, 1, 2])).sum()
        bad_topic = (~df["topic"].isin([0, 1, 2, 3])).sum()
        invalid += bad_sent + bad_topic
    if invalid == 0:
        r.pass_("All sentiment∈{0,1,2}, topic∈{0,1,2,3}")
    else:
        r.fail_(f"{invalid} invalid labels found")
    results.append(r)

    # I-6: ID duy nhất
    r = ValidationResult("I-6", "Unique IDs per file")
    dup_ids = 0
    for s, df in splits.items():
        dups = df["id"].duplicated().sum()
        dup_ids += dups
    if dup_ids == 0:
        r.pass_("All IDs unique within each file")
    else:
        r.fail_(f"{dup_ids} duplicate IDs found")
    results.append(r)

    # I-7: Cross-split clean (no text overlap)
    r = ValidationResult("I-7", "No cross-split text overlap")
    train_texts = set(splits["train"]["text"].tolist())
    dev_texts = set(splits["dev"]["text"].tolist())
    test_texts = set(splits["test"]["text"].tolist())
    overlap_td = len(train_texts & dev_texts)
    overlap_tt = len(train_texts & test_texts)
    overlap_dt = len(dev_texts & test_texts)
    total_overlap = overlap_td + overlap_tt + overlap_dt
    if total_overlap == 0:
        r.pass_("0 overlapping texts between splits")
    else:
        r.fail_(
            f"Overlaps: train∩dev={overlap_td}, "
            f"train∩test={overlap_tt}, dev∩test={overlap_dt}"
        )
    results.append(r)

    # I-8: Encoding đúng (UTF-8 readable)
    r = ValidationResult("I-8", "UTF-8 encoding correct")
    encoding_ok = True
    for s in ["train", "dev", "test"]:
        filepath = os.path.join(PROCESSED_DIR, f"{s}.csv")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                f.read()
        except UnicodeDecodeError as e:
            encoding_ok = False
            r.fail_(f"{s}.csv: {e}")
            break
    if encoding_ok:
        r.pass_("All CSVs readable as UTF-8")
    results.append(r)

    # I-9: Dev/Test không đổi số mẫu so với raw
    r = ValidationResult("I-9", "Dev/Test sample count unchanged from raw")
    dev_ok = len(splits["dev"]) == 1583
    test_ok = len(splits["test"]) == 3166
    if dev_ok and test_ok:
        r.pass_("dev=1583, test=3166 (match raw)")
    else:
        r.fail_(f"dev={len(splits['dev'])}, test={len(splits['test'])}")
    results.append(r)

    # I-10: Conflict đã xóa
    r = ValidationResult("I-10", "Annotation conflict removed from processed train")
    conflict_in_train = splits["train"][
        splits["train"]["text"] == CONFLICT_TEXT
    ]
    if len(conflict_in_train) == 0:
        r.pass_("Conflict text not found in processed train")
    else:
        r.fail_(f"Conflict text still present: {len(conflict_in_train)} rows")
    results.append(r)

    # I-11: Phân bố dev/test giữ nguyên
    r = ValidationResult("I-11", "Dev/Test label distribution unchanged from raw")
    dist_ok = True
    problems = []

    dev_sent = (
        splits["dev"]["sentiment"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    dev_sent_str = {str(k): v for k, v in dev_sent.items()}
    if dev_sent_str != EXPECTED_DEV_SENTIMENT:
        dist_ok = False
        problems.append(f"dev sentiment: expected {EXPECTED_DEV_SENTIMENT}, got {dev_sent_str}")

    dev_topic = (
        splits["dev"]["topic"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    dev_topic_str = {str(k): v for k, v in dev_topic.items()}
    if dev_topic_str != EXPECTED_DEV_TOPIC:
        dist_ok = False
        problems.append(f"dev topic: expected {EXPECTED_DEV_TOPIC}, got {dev_topic_str}")

    test_sent = (
        splits["test"]["sentiment"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    test_sent_str = {str(k): v for k, v in test_sent.items()}
    if test_sent_str != EXPECTED_TEST_SENTIMENT:
        dist_ok = False
        problems.append(f"test sentiment: expected {EXPECTED_TEST_SENTIMENT}, got {test_sent_str}")

    test_topic = (
        splits["test"]["topic"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    test_topic_str = {str(k): v for k, v in test_topic.items()}
    if test_topic_str != EXPECTED_TEST_TOPIC:
        dist_ok = False
        problems.append(f"test topic: expected {EXPECTED_TEST_TOPIC}, got {test_topic_str}")

    if dist_ok:
        r.pass_("Dev and test label distributions match raw data")
    else:
        r.fail_("; ".join(problems))
    results.append(r)

    return results


def run_quality_checks() -> list:
    """5 Quality checks — cảnh báo."""
    results = []

    splits = {s: load_processed_csv(s) for s in ["train", "dev", "test"]}

    # Q-1: Không còn emoji acronym
    r = ValidationResult("Q-1", "No emoji acronyms remaining in text_clean")
    acronym_found = 0
    for s, df in splits.items():
        for acronym in ALL_ACRONYM_KEYS:
            # Chỉ kiểm tra acronym thực sự là emoji, bỏ qua các ký tự thường
            # (doubledot→":", vdotv→"v.v", etc. đã chuyển về ký tự gốc)
            count = df["text_clean"].str.contains(acronym, regex=False).sum()
            if count > 0:
                acronym_found += count
    if acronym_found == 0:
        r.pass_("0 acronyms found in text_clean")
    else:
        r.fail_(f"{acronym_found} acronym occurrences still present")
    results.append(r)

    # Q-2: Không còn khoảng trắng thừa
    r = ValidationResult("Q-2", "No double spaces in text_clean")
    double_space = 0
    for s, df in splits.items():
        ds = df["text_clean"].str.contains("  ", regex=False).sum()
        double_space += ds
    if double_space == 0:
        r.pass_("0 double-space occurrences")
    else:
        r.fail_(f"{double_space} rows with double spaces")
    results.append(r)

    # Q-3: text_lower nhất quán
    r = ValidationResult("Q-3", "text_lower == text_clean.lower() for all rows")
    mismatch = 0
    for s, df in splits.items():
        mm = (df["text_lower"] != df["text_clean"].str.lower()).sum()
        mismatch += mm
    if mismatch == 0:
        r.pass_("100% consistent")
    else:
        r.fail_(f"{mismatch} mismatches")
    results.append(r)

    # Q-4: text vs text_clean khác nhau (info)
    r = ValidationResult("Q-4", "Text changed by preprocessing")
    changed_counts = {}
    for s, df in splits.items():
        changed = (df["text"] != df["text_clean"]).sum()
        changed_counts[s] = int(changed)
    total_changed = sum(changed_counts.values())
    r.pass_(
        f"{total_changed} sentences changed "
        f"(train={changed_counts['train']}, "
        f"dev={changed_counts['dev']}, "
        f"test={changed_counts['test']})"
    )
    results.append(r)

    # Q-5: Emoji Unicode hiện diện
    r = ValidationResult("Q-5", "Emoji Unicode present in text_clean")
    emoji_chars = [
        "\U0001F642",  # 🙂
        "\U0001F61E",  # 😞
        "\U0001F604",  # 😄
        "\U0001F60A",  # 😊
        "\U0001F606",  # 😆
        "❤️",  # ❤️ (chú ý: 2 codepoints ❤ + FE0F)
        "❤",           # ❤ (1 codepoint, dự phòng)
    ]
    found_any = False
    emoji_total = 0
    for s, df in splits.items():
        for emoji in emoji_chars:
            count = df["text_clean"].str.contains(emoji, regex=False).sum()
            emoji_total += count
            if count > 0:
                found_any = True
    if found_any:
        r.pass_(f"Emoji Unicode found ({emoji_total} total occurrences)")
    else:
        r.fail_("No emoji Unicode found — replacement may have failed")
    results.append(r)

    return results


def main():
    """Chạy tất cả validation checks."""
    print("=" * 60)
    print("VALIDATION — PREPROCESSING OUTPUT")
    print("Đề tài #2 — Theo PREPROCESSING_PLAN.md đã duyệt")
    print("=" * 60)

    # === Integrity checks ===
    print("\n=== INTEGRITY CHECKS (11 checks — PHẢI PASS) ===\n")
    integrity_results = run_integrity_checks()
    for r in integrity_results:
        print(f"  {r}")

    integrity_passed = sum(1 for r in integrity_results if r.passed)
    integrity_total = len(integrity_results)

    # === Quality checks ===
    print("\n=== QUALITY CHECKS (5 checks — cảnh báo) ===\n")
    quality_results = run_quality_checks()
    for r in quality_results:
        print(f"  {r}")

    quality_passed = sum(1 for r in quality_results if r.passed)
    quality_total = len(quality_results)

    # === Tổng kết ===
    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print(f"  Integrity: {integrity_passed}/{integrity_total} PASSED")
    print(f"  Quality:   {quality_passed}/{quality_total} PASSED")

    all_integrity_pass = integrity_passed == integrity_total
    all_quality_pass = quality_passed == quality_total

    if all_integrity_pass and all_quality_pass:
        print("\n  Status: ✅ ALL CHECKS PASSED — READY FOR MODEL TRAINING")
    elif all_integrity_pass:
        print("\n  Status: ⚠️ INTEGRITY OK, some quality warnings")
    else:
        print("\n  Status: ❌ INTEGRITY CHECKS FAILED — DO NOT PROCEED")
        failed = [r for r in integrity_results if not r.passed]
        print("\n  Failed checks:")
        for r in failed:
            print(f"    {r}")

    print("=" * 60)

    return all_integrity_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
