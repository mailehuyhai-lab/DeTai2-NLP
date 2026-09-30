"""Run the full UIT-VSFC EDA end-to-end.

Reads ONLY data/processed/{train,dev,test}.csv (+ label_maps.json).
Writes tables, figures and emoji_summary.json under results/eda/.

Usage:  python src/eda/run_eda.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda import analyze as A
from src.eda import plots as P

TABLES = A.RESULTS_DIR / "tables"
FIGS = A.RESULTS_DIR / "figures"
SUMM = A.RESULTS_DIR / "summaries"


def main() -> int:
    for d in (TABLES, FIGS, SUMM):
        d.mkdir(parents=True, exist_ok=True)

    sent_map, topic_map = A.load_label_maps()
    data = A.load_all()
    splits = ["train", "dev", "test"]

    # ---- 1. Label distributions -------------------------------------
    print("[1] Label distributions")
    sent_dist_all = {s: A.label_distribution(data[s], "sentiment", sent_map)
                     for s in splits}
    topic_dist_all = {s: A.label_distribution(data[s], "topic", topic_map)
                      for s in splits}

    sent_dist_all["train"].to_csv(TABLES / "sentiment_distribution.csv", index=False)
    topic_dist_all["train"].to_csv(TABLES / "topic_distribution.csv", index=False)

    # combined long-format distribution across splits (for the report)
    comb = []
    for s in splits:
        for task, dist in (("sentiment", sent_dist_all[s]),
                           ("topic", topic_dist_all[s])):
            for _, r in dist.iterrows():
                comb.append({"split": s, "task": task, **r.to_dict()})
    pd.DataFrame(comb).to_csv(TABLES / "split_distribution_summary.csv", index=False)

    P.grouped_label_bars(sent_dist_all, sent_map,
                         "Sentiment distribution — train/dev/test",
                         FIGS / "sentiment_distribution_all.png")
    P.grouped_label_bars(topic_dist_all, topic_map,
                         "Topic distribution — train/dev/test",
                         FIGS / "topic_distribution_all.png")

    # imbalance ratios
    imb = {
        "sentiment": {s: A.imbalance_ratio(sent_dist_all[s]) for s in splits},
        "topic": {s: A.imbalance_ratio(topic_dist_all[s]) for s in splits},
    }
    with open(SUMM / "imbalance_summary.json", "w", encoding="utf-8") as f:
        json.dump(imb, f, ensure_ascii=False, indent=2)

    # ---- 2. Text length ----------------------------------------------
    print("[2] Text length")
    len_stats = {s: A.text_length_stats(data[s]) for s in splits}
    # flat summary table
    len_rows = []
    for s in splits:
        st = len_stats[s]
        row = {"split": s, "n_samples": st["n_samples"]}
        for metric in ("chars", "words", "phobert_tokens"):
            if st.get(metric):
                for k in ("min", "max", "mean", "median", "p90", "p95", "p99"):
                    row[f"{metric}_{k}"] = st[metric][k]
        row["truncated_gt_32"] = st.get("phobert_truncated_gt_32")
        row["truncated_gt_32_pct"] = st.get("phobert_truncated_gt_32_pct")
        len_rows.append(row)
    pd.DataFrame(len_rows).to_csv(TABLES / "text_length_summary.csv", index=False)

    P.length_histogram({s: data[s]["text_clean"].str.split().str.len() for s in splits},
                       "Text length distribution (words/comment)",
                       FIGS / "text_length_distribution.png")

    # ---- 3. Keywords --------------------------------------------------
    print("[3] Keywords")
    # overall = combined train+dev+test for a corpus-level view
    corpus = pd.concat([data[s] for s in splits], ignore_index=True)
    kw_overall = A.keyword_frequency(corpus, top_n=30)
    kw_overall.to_csv(TABLES / "keyword_frequency.csv", index=False)
    P.top_keywords_bar(kw_overall, "Top keywords — overall",
                       FIGS / "top_keywords_overall.png")

    kw_sent = A.keyword_frequency_by(corpus, "sentiment", sent_map, top_n=20)
    P.top_keywords_grid(kw_sent, "Top keywords by sentiment",
                        FIGS / "top_keywords_sentiment.png")

    kw_topic = A.keyword_frequency_by(corpus, "topic", topic_map, top_n=20)
    P.top_keywords_grid(kw_topic, "Top keywords by topic",
                        FIGS / "top_keywords_topic.png")

    # per-class keyword table
    kw_rows = []
    for task, by in (("sentiment", kw_sent), ("topic", kw_topic)):
        for name, d in by.items():
            for _, r in d.iterrows():
                kw_rows.append({"task": task, "class": name,
                                "keyword": r["keyword"], "count": r["count"]})
    pd.DataFrame(kw_rows).to_csv(TABLES / "keyword_frequency_by_class.csv", index=False)

    # ---- 4. Sentiment x Topic -----------------------------------------
    print("[4] Sentiment x Topic")
    ct, pct = A.sentiment_topic_crosstab(corpus, sent_map, topic_map)
    ct.to_csv(TABLES / "sentiment_topic_counts.csv")
    pct.to_csv(TABLES / "sentiment_topic_row_pct.csv")
    P.crosstab_heatmap(ct, "Sentiment × Topic counts (all splits)",
                       FIGS / "sentiment_topic_heatmap.png")

    # ---- 5. Emoji ------------------------------------------------------
    print("[5] Emoji")
    emoji = {s: A.emoji_stats(data[s]) for s in splits}
    with open(SUMM / "emoji_summary.json", "w", encoding="utf-8") as f:
        json.dump(emoji, f, ensure_ascii=False, indent=2)
    if emoji["train"]["top_emojis"]:
        P.emoji_bar(emoji["train"]["top_emojis"],
                    "Top emojis — train split",
                    FIGS / "emoji_frequency.png")

    print("\nEDA complete. Artifacts under results/eda/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
