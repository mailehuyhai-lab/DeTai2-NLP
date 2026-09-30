# EDA RESULT — UIT-VSFC

**Đề tài #2** — Sentiment + Topic classification, Trường Đại học Nha Trang, Củ Chi 2026
**Step:** Exploratory Data Analysis (Mức 1) — **read-only** on `data/processed/`
**Date:** 2026-09-30

> The UIT-VSFC dataset does not provide a timestamp field, therefore temporal
> trend analysis cannot be performed from the original data. This limitation is
> recorded for the later Dashboard step.

---

## 1. Objective

Profile the preprocessed UIT-VSFC corpus to answer: label distribution, class
imbalance, text-length behaviour (incl. truncation under the transformer's
`max_length=32`), frequent keywords, sentiment×topic joint structure, and
train/dev/test consistency — to inform (not decide) downstream modelling.

EDA is strictly **read-only**: it reads `data/processed/{train,dev,test}.csv` and
`label_maps.json`. No file under `data/raw/`, `data/processed/`, `models/`,
`results/baseline/`, or `results/transformer/` was modified.

## 2. Dataset Overview

| Split | Samples |
|-------|---------|
| Train | 11,424  |
| Dev   | 1,583   |
| Test  | 3,166   |
| **Total** | **16,173** |

Columns used: `id`, `text`, `text_clean`, `text_lower`, `sentiment` (0–2),
`topic` (0–3). No `*_name` columns exist — names are mapped from
`data/processed/label_maps.json`. No nulls in any field EDA consumes.

Label maps:
- Sentiment: 0=Negative, 1=Neutral, 2=Positive
- Topic: 0=Lecturer, 1=Training_program, 2=Facility, 3=Others

## 3. Sentiment Distribution

Per split (count / %), `tables/split_distribution_summary.csv`:

| Split | Negative | Neutral | Positive |
|-------|----------|---------|----------|
| Train | 5,324 (46.60%) | 458 (4.01%) | 5,642 (49.39%) |
| Dev   | 705 (44.54%)   | 73 (4.61%)  | 805 (50.85%)   |
| Test  | 1,409 (44.50%) | 167 (5.27%) | 1,590 (50.22%) |

Figure: `figures/sentiment_distribution_all.png`.

Negative and Positive are roughly balanced (~45% / ~50%); **Neutral is the clear
minority at ~4–5%**.

## 4. Topic Distribution

| Split | Lecturer | Training_program | Facility | Others |
|-------|----------|------------------|----------|--------|
| Train | 8,164 (71.46%) | 2,201 (19.27%) | 497 (4.35%) | 562 (4.92%) |
| Dev   | 1,151 (72.71%) | 267 (16.87%)   | 70 (4.42%)  | 95 (6.00%)  |
| Test  | 2,290 (72.33%) | 572 (18.07%)   | 145 (4.58%) | 159 (5.02%) |

Figure: `figures/topic_distribution_all.png`.

**Lecturer dominates (~71–73%)**; Training_program is a moderate second
(~17–19%); **Facility and Others are small (~4–6% each)**.

## 5. Class Imbalance

Majority/minority ratios (`summaries/imbalance_summary.json`):

| Task | Split | Majority | Minority | Ratio |
|------|-------|----------|----------|-------|
| Sentiment | Train | Positive 5,642 | **Neutral 458** | **12.3×** |
| Sentiment | Dev   | Positive 805   | Neutral 73   | 11.0× |
| Sentiment | Test  | Positive 1,590 | Neutral 167  | 9.5×  |
| Topic | Train | Lecturer 8,164 | **Facility 497** | **16.4×** |
| Topic | Dev   | Lecturer 1,151 | Facility 70    | 16.4× |
| Topic | Test  | Lecturer 2,290 | Facility 145   | 15.8× |

**Minority classes:** Neutral (sentiment); Facility and Others (topic). This is
an **imbalance description, not a data defect**. Potential effect: a model can
reach high accuracy while under-serving the rare classes, so **macro-F1 and
per-class recall are the informative metrics** (consistent with the weighted-loss
choice already used in the transformer step).

## 6. Text Length Analysis

Computed on `text_clean` — characters, whitespace words, and PhoBERT BPE tokens
(`tables/text_length_summary.csv`):

| Split | Metric | min | max | mean | median | p90 | p95 | p99 |
|-------|--------|-----|-----|------|--------|-----|-----|-----|
| Train | chars  | 4   | 660 | 58.9 | 47     | 110 | 141 | 219 |
| Train | words  | 2   | 159 | 14.3 | 11     | 26  | 33  | 52  |
| Train | tokens | 2   | 161 | 14.6 | 12     | 27  | 34  | 53  |
| Dev   | words  | 2   | 161 | 13.7 | 11     | 25  | 31  | 46  |
| Dev   | tokens | 2   | 162 | 13.9 | 11     | 25  | 33  | 50  |
| Test  | words  | 2   | 98  | 14.2 | 11     | 26  | 34  | 52  |
| Test  | tokens | 2   | 100 | 14.5 | 11     | 27  | 35  | 53  |

Figure: `figures/text_length_distribution.png` (right-skewed; most comments are
short, long tail to ~160 words/tokens).

**Truncation under the transformer's `max_length = 32`:**

| Split | tokens > 32 | % truncated |
|-------|-------------|-------------|
| Train | 678  | **5.93%** |
| Dev   | 81   | 5.12%   |
| Test  | 201  | 6.35%   |

So **~94% of samples fit entirely within 32 tokens**; the ~6% truncated lose tail
text only. This is a measured coverage figure — `max_length` itself is **not
changed** by this EDA.

## 7. Keyword Analysis

Method: lowercase `text_clean`, regex tokenise to word tokens, drop a fixed
embedded Vietnamese stopword list + tokens shorter than 2 chars
(`analyze.VIETNAMESE_STOPWORDS`). No external stopword download; no TF-IDF
vocabulary reuse. Frequency = raw corpus count. Reproducible (validated V9).

Overall top keywords (`tables/keyword_frequency.csv`,
`figures/top_keywords_overall.png`): viên, giảng, dạy, sinh, học, bài, tình,
nhiệt, hiểu, dễ, tập, thực, môn, tâm, thức, kiến, tận, hành, lớp, cần, …

Top per-class (top ~8; full top-20 in `tables/keyword_frequency_by_class.csv`):

| Class | Frequent keywords |
|-------|-------------------|
| Negative | viên, học, giảng, sinh, bài, thực, dạy, tập |
| Neutral  | học, viên, bài, dạy, kiến, tập, giảng, sinh |
| Positive | viên, giảng, dạy, **tình, nhiệt**, sinh, **dễ, hiểu** |
| Lecturer | viên, giảng, dạy, sinh, tình, nhiệt, bài, học |
| Training_program | học, bài, thực, môn, viên, sinh, tập, hành |
| Facility | **phòng, máy**, học, thực, hành, thiết, cần, **chiếu** |
| Others | học, viên, sinh, kiến, môn, thực, lớp, hiểu |

Figures: `top_keywords_overall.png`, `top_keywords_sentiment.png`,
`top_keywords_topic.png`.

Notes:
- Many "keywords" are **syllable fragments** of multi-syllable Vietnamese words
  (viên←{giảng viên, sinh viên}, giảng←{giảng viên}, nhiệt←{nhiệt tình}). This is
  expected — `text_clean` is whitespace/syllable text — and is why the domain
  signal still emerges (e.g., Facility surfaces *phòng, máy, chiếu* = room/machine/
  projector).
- Positive skews toward *tình/nhiệt (nhiệt tình)* and *dễ/hiểu (dễ hiểu)* — the
  usual praise phrasing; Negative/Neutral share more generic terms. These are
  **co-occurrence frequencies only — not causal** drivers of the label.

## 8. Sentiment × Topic Analysis

Contingency over the full corpus (`tables/sentiment_topic_counts.csv`,
`sentiment_topic_row_pct.csv`, `figures/sentiment_topic_heatmap.png` — log-scaled
colour, raw-count annotations):

| Sentiment | Lecturer | Training_program | Facility | Others |
|-----------|----------|------------------|----------|--------|
| Negative  | 4,104 | 2,328 | 681 | 325 |
| Neutral   | 292   | 162   | 13  | 231 |
| Positive  | 7,209 | 550   | 18  | 260 |

Row-percentage (share of each sentiment across topics):

| Sentiment | Lecturer | Training_program | Facility | Others |
|-----------|----------|------------------|----------|--------|
| Negative  | 55.2% | **31.3%** | 9.2% | 4.4% |
| Neutral   | 41.8% | 23.2% | 1.9% | **33.1%** |
| Positive  | **89.7%** | 6.8% | 0.2% | 3.2% |

Descriptive patterns (statistics only, no causal claim):
- **Positive feedback concentrates heavily on Lecturer** (≈90% of positives).
- **Negative feedback is spread** across Lecturer (55%) and Training_program
  (31%), and is the only sentiment with a meaningful Facility share (9.2%) —
  complaints about facilities are almost always negative.
- **Neutral feedback is disproportionately in Others** (≈33% of neutrals) —
  neutral comments are often off-topic/uncategorised.
- Facility is almost never Positive (18 / 897 total positives ≈ 0.2%).

## 9. Train / Dev / Test Consistency

`tables/split_distribution_summary.csv`, `tables/text_length_summary.csv`:

- **Sentiment** — Neg/Neu/Pos percentages are within ~1–2 points across splits
  (e.g., Positive 49.4 / 50.9 / 50.2; Neutral 4.0 / 4.6 / 5.3). Consistent.
- **Topic** — Lecturer 71.5 / 72.7 / 72.3; Facility 4.4 / 4.4 / 4.6; Others 4.9 /
  6.0 / 5.0. Consistent, slight Others uptick in dev.
- **Text length** — mean words 14.3 / 13.7 / 14.2; median 11 across all; p95
  tokens 34 / 33 / 35. Essentially identical.

No notable distribution difference is observed descriptively between the three splits. The label and text-length distributions are broadly consistent across train/dev/test. Report only; nothing to correct.

## 10. Emoji / Special Text Signals

Preprocessing already mapped emoji acronyms → Unicode; EDA counts the Unicode
pictographs present (`summaries/emoji_summary.json`, `figures/emoji_frequency.png`):

| Split | n_samples | with emoji | % |
|-------|-----------|------------|---|
| Train | 11,424 | 84  | **0.74%** |
| Dev   | 1,583  | 10  | 0.63% |
| Test  | 3,166  | 31  | 0.98% |

Top emojis (train): 🙂 (27), 😄 (12), 😞 (11), 😏 (9), ❤ (7), 😆 (5).

**Emoji are rare (~0.6–1.0% of comments).** They carry a mild positive/negative
sentiment cue where present but occur too infrequently to drive the task. No
emoji were removed or re-processed — EDA is read-only.

## 11. Findings Relevant to Modeling

These are facts the EDA directly supports (no performance claims):

- **Sentiment is imbalanced** — Neutral ~4–5% (≈12× majority/minority on train).
  Accuracy alone would mask Neutral; macro-F1 / per-class recall needed.
- **Topic is imbalanced** — Lecturer ~71–73%; Facility + Others ~4–6% each
  (~16×). Same implication.
- **Text is short** — median ≈11 words / 12 tokens; ~94% fit `max_length=32`;
  ~6% are truncated at the tail. Truncation affects a small minority only.
- **Splits are consistent** — label and length distributions match across
  train/dev/test, so dev is a reasonable model-selection proxy and test a fair
  final check.
- **Sentiment and topic are not independent** — positives concentrate on
  Lecturer (~90%), neutrals concentrate on Others (~33%); Facility is
  overwhelmingly negative. This joint distribution is a relevant descriptive finding for later analysis.
- **Keywords are domain-specific** — Facility → phòng/máy/chiếu;
  Training_program → môn/bài/thực hành; Lecturer → giảng viên/nhiệt tình/dễ hiểu.
  Frequencies, not causal features.
- **Emoji signal is sparse** (~1%) — usable as a feature only if treated as a
  rare indicator, not a primary cue.
- No timestamp field — **temporal analysis is not possible** (Dashboard will
  surface this limitation).

## 12. Limitations

- EDA is descriptive only — it cannot establish which features *cause* a label,
  nor predict model performance.
- Keyword tokens are syllable-level; compound-word meaning must be read across
  fragments. No word-segmenter was applied (consistent with the PhoBERT input).
- The embedded Vietnamese stopword list is fixed and domain-curated; it is not
  an authoritative external list, but it is reproducible and versioned here.
- Emoji figure labels use Unicode names because the default font lacks emoji
  glyphs; counts are unaffected.
- `max_length=32` analysis reports truncation coverage only; it does not
  re-tune that hyperparameter.

## 13. Generated Artifacts

**Code** — `src/eda/`:
`__init__.py`, `analyze.py`, `plots.py`, `run_eda.py`, `validate.py`

**Tables** — `results/eda/tables/`:
`split_distribution_summary.csv`, `text_length_summary.csv`,
`sentiment_distribution.csv`, `topic_distribution.csv`,
`sentiment_topic_counts.csv`, `sentiment_topic_row_pct.csv`,
`keyword_frequency.csv`, `keyword_frequency_by_class.csv`

**Figures** — `results/eda/figures/`:
`sentiment_distribution_all.png`, `topic_distribution_all.png`,
`text_length_distribution.png`, `top_keywords_overall.png`,
`top_keywords_sentiment.png`, `top_keywords_topic.png`,
`sentiment_topic_heatmap.png`, `emoji_frequency.png`

**Summaries** — `results/eda/summaries/`:
`emoji_summary.json`, `imbalance_summary.json`

## 14. EDA Status

Validation `src/eda/validate.py` — **ALL V1–V10 PASSED**:
inputs exist; row counts 11424/1583/3166; no nulls; valid labels; inputs
unmodified (MD5 stable); 9/9 raw MD5 unchanged; all 7 required tables present and
non-empty; all 8 figures generated; keyword stats deterministic; sentiment×topic
total = 16,173.

**EDA STATUS: READY FOR REVIEW**
