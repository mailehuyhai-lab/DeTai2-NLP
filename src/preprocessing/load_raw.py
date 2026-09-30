"""
load_raw.py — Đọc và ghép sents.txt + sentiments.txt + topics.txt từ data/raw/

Đề tài #2: Phân tích cảm xúc và chủ đề từ phản hồi người dùng
Theo PREPROCESSING_PLAN.md đã duyệt 2026-09-29

NGUYÊN TẮC: Chỉ ĐỌC từ data/raw/. KHÔNG sửa bất kỳ file nào.
"""

import os
import hashlib
import pandas as pd


# MD5 checksums tham chiếu từ PREPROCESSING_PLAN.md (mục 2.4)
RAW_CHECKSUMS = {
    "train/sents.txt": "dd4d13bee582f9120f5dcfa5d126662e",
    "train/sentiments.txt": "b807001a6c4c573cdd3fe236fa06add4",
    "train/topics.txt": "337e2f21d16e51b3ee4d1013e018458c",
    "dev/sents.txt": "4b01b2b3df63d6dd0c64c0e9c1f4c763",
    "dev/sentiments.txt": "bb87e5ee63082ea3c438f130a2a227aa",
    "dev/topics.txt": "11e9e0ea3beb1aafbdddc424b0696407",
    "test/sents.txt": "800f706107d9e634563848a681566c34",
    "test/sentiments.txt": "e125fa119d7c1c0a21337a9e80b70f29",
    "test/topics.txt": "224c091149ad257ab72f742cf545ab06",
}


def compute_md5(filepath: str) -> str:
    """Tính MD5 checksum của file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_checksums(raw_dir: str) -> dict:
    """
    Kiểm tra MD5 checksum của 9 file raw.
    Trả về dict: {relative_path: {"expected": ..., "actual": ..., "match": bool}}
    Nếu bất kỳ file nào không khớp → raise lỗi.
    """
    results = {}
    all_match = True

    for rel_path, expected_md5 in RAW_CHECKSUMS.items():
        filepath = os.path.join(raw_dir, rel_path)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Raw file not found: {filepath}")

        actual_md5 = compute_md5(filepath)
        match = actual_md5 == expected_md5
        results[rel_path] = {
            "expected": expected_md5,
            "actual": actual_md5,
            "match": match,
        }
        if not match:
            all_match = False

    if not all_match:
        mismatched = [p for p, r in results.items() if not r["match"]]
        raise RuntimeError(
            f"RAW DATA INTEGRITY VIOLATION! "
            f"Files with changed checksums: {mismatched}. "
            f"DỪNG NGAY — data/raw/ đã bị thay đổi!"
        )

    return results


def _read_lines(filepath: str, encoding: str = "utf-8") -> list:
    """Đọc file, trả về list các dòng đã strip."""
    with open(filepath, "r", encoding=encoding) as f:
        return [line.strip() for line in f.readlines()]


def load_split(raw_dir: str, split: str) -> pd.DataFrame:
    """
    Đọc 1 split (train/dev/test) từ data/raw/.

    Args:
        raw_dir: Đường dẫn tới data/raw/
        split: "train", "dev", hoặc "test"

    Returns:
        DataFrame với cột: id, text, sentiment, topic, split
    """
    split_dir = os.path.join(raw_dir, split)

    # Đọc 3 file
    sents = _read_lines(os.path.join(split_dir, "sents.txt"), encoding="utf-8")
    sentiments = _read_lines(os.path.join(split_dir, "sentiments.txt"), encoding="ascii")
    topics = _read_lines(os.path.join(split_dir, "topics.txt"), encoding="ascii")

    # Validate số dòng trước khi ghép
    n_sents = len(sents)
    n_sentiments = len(sentiments)
    n_topics = len(topics)

    assert n_sents == n_sentiments == n_topics, (
        f"Line count mismatch in {split}: "
        f"sents={n_sents}, sentiments={n_sentiments}, topics={n_topics}"
    )

    # Ghép thành DataFrame
    df = pd.DataFrame({
        "id": [f"{split}_{i:05d}" for i in range(n_sents)],
        "text": sents,
        "sentiment": [int(s) for s in sentiments],
        "topic": [int(t) for t in topics],
        "split": split,
    })

    return df


def load_all_splits(raw_dir: str) -> dict:
    """
    Đọc tất cả 3 split từ data/raw/.

    Args:
        raw_dir: Đường dẫn tới data/raw/

    Returns:
        dict: {"train": DataFrame, "dev": DataFrame, "test": DataFrame}
    """
    # Kiểm tra checksum TRƯỚC khi đọc
    verify_raw_checksums(raw_dir)

    result = {}
    for split in ["train", "dev", "test"]:
        result[split] = load_split(raw_dir, split)

    return result
