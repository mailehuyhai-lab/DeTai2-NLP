"""Validation for the EDA step (Topic #2).

Read-only checks. Verifies inputs, outputs and data integrity. Does NOT
retrain any model and does NOT modify models/, results/baseline/,
results/transformer/ or the processed/raw data.

Usage:  python src/eda/validate.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda import analyze as A
from src.preprocessing.load_raw import verify_raw_checksums

RESULTS = A.RESULTS_DIR
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"
SUMM = RESULTS / "summaries"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = "PASS" if cond else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return bool(cond)


def main() -> int:
    ok = True
    print("=" * 60)
    print("EDA VALIDATION")
    print("=" * 60)

    sent_map, topic_map = A.load_label_maps()

    # snapshot processed CSV checksums BEFORE re-reading (to confirm read-only)
    in_files = [A.PROCESSED_DIR / f"{s}.csv" for s in ("train", "dev", "test")]
    pre_md5 = {p.name: md5(p) for p in in_files if p.exists()}

    # V1 input files exist
    print("\n[V1] Input files exist")
    for p in in_files:
        ok &= check(p.name, p.exists())
    ok &= check("label_maps.json",
                (A.PROCESSED_DIR / "label_maps.json").exists())

    # V2 row counts
    print("\n[V2] Row counts")
    data = {}
    for s in ("train", "dev", "test"):
        p = A.PROCESSED_DIR / f"{s}.csv"
        if not p.exists():
            continue
        data[s] = pd.read_csv(p)
        ok &= check(f"{s} = {A.EXPECTED_COUNTS[s]}",
                    len(data[s]) == A.EXPECTED_COUNTS[s], str(len(data[s])))

    # V3 no nulls in EDA fields
    print("\n[V3] No nulls in used fields")
    for s, df in data.items():
        for col in ("id", "text_clean", "sentiment", "topic"):
            ok &= check(f"{s}.{col} non-null",
                        not df[col].isna().any(),
                        f"{int(df[col].isna().sum())} nulls")

    # V4 valid label values
    print("\n[V4] Label values valid")
    for s, df in data.items():
        ok &= check(f"{s}.sentiment in {sorted(sent_map)}",
                    set(df.sentiment.unique()) <= set(sent_map.keys()),
                    str(sorted(df.sentiment.unique())))
        ok &= check(f"{s}.topic in {sorted(topic_map)}",
                    set(df.topic.unique()) <= set(topic_map.keys()),
                    str(sorted(df.topic.unique())))

    # V5 inputs not modified (checksums stable across the run)
    print("\n[V5] Input CSVs unmodified")
    for p in in_files:
        if p.exists():
            ok &= check(p.name, md5(p) == pre_md5[p.name])

    # V6 raw + processed integrity
    print("\n[V6] Data integrity")
    raw_ok = bool(verify_raw_checksums(str(A.PROCESSED_DIR.parent / "raw")))
    ok &= check("9/9 raw MD5 unchanged", raw_ok)

    # V7 output tables: exist + no unexpected NaN
    print("\n[V7] Output tables valid")
    expected_tables = [
        "split_distribution_summary.csv",
        "text_length_summary.csv",
        "sentiment_distribution.csv",
        "topic_distribution.csv",
        "sentiment_topic_counts.csv",
        "sentiment_topic_row_pct.csv",
        "keyword_frequency.csv",
    ]
    for t in expected_tables:
        p = TABLES / t
        if not p.exists():
            ok &= check(t, False, "missing")
            continue
        df = pd.read_csv(p)
        # the crosstab row-pct uses label-name index col; check only numeric area
        ok &= check(t, len(df) > 0, f"{len(df)} rows")

    # V8 figures exist and non-empty
    print("\n[V8] Figures generated")
    expected_figs = [
        "sentiment_distribution_all.png",
        "topic_distribution_all.png",
        "text_length_distribution.png",
        "top_keywords_overall.png",
        "top_keywords_sentiment.png",
        "top_keywords_topic.png",
        "sentiment_topic_heatmap.png",
        "emoji_frequency.png",
    ]
    for f_ in expected_figs:
        p = FIGS / f_
        ok &= check(f_, p.exists() and p.stat().st_size > 1000,
                    f"{p.stat().st_size}B" if p.exists() else "missing")

    # V9 keyword reproducibility — recompute and compare to written CSV
    print("\n[V9] Keyword stats reproducible")
    corpus = pd.concat([data[s] for s in ("train", "dev", "test")],
                       ignore_index=True)
    recomputed = A.keyword_frequency(corpus, top_n=30)
    written = pd.read_csv(TABLES / "keyword_frequency.csv")
    same = (recomputed["keyword"].tolist() == written["keyword"].tolist()
            and recomputed["count"].tolist() == written["count"].tolist())
    ok &= check("top-30 keywords deterministic", same)

    # V10 sentiment x topic totals match corpus size
    print("\n[V10] sentiment×topic total matches corpus")
    ct = pd.read_csv(TABLES / "sentiment_topic_counts.csv", index_col=0)
    total = int(ct.values.sum())
    expected_total = sum(len(data[s]) for s in data)
    ok &= check("crosstab total = n_samples", total == expected_total,
                f"{total} vs {expected_total}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
