"""EDA analysis utilities for UIT-VSFC.

All functions are read-only: they consume the processed CSVs and produce
statistics DataFrames / dicts. Nothing here writes to or mutates the input.

Constants:
    PROCESSED_DIR: data/processed/
    SENTIMENT_MAP / TOPIC_MAP: loaded from data/processed/label_maps.json
    VIETNAMESE_STOPWORDS: fixed, embedded list (documented, reproducible).
    MAX_LENGTH_USED: the transformer max_length=32, for reporting truncation
                     coverage. This module does NOT change it.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "eda"

# Transformer max_length used in the PhoBERT step. Reported only — never changed.
MAX_LENGTH_USED = 32

RANDOM_STATE = 42

EXPECTED_COUNTS = {"train": 11424, "dev": 1583, "test": 3166}


def load_label_maps() -> tuple[dict, dict]:
    """Load label id -> name maps from data/processed/label_maps.json."""
    with open(PROCESSED_DIR / "label_maps.json", encoding="utf-8") as f:
        maps = json.load(f)
    sent = {int(k): v for k, v in maps["sentiment"].items()}
    topic = {int(k): v for k, v in maps["topic"].items()}
    return sent, topic


def load_split(split: str) -> pd.DataFrame:
    """Read one processed split. Read-only."""
    return pd.read_csv(PROCESSED_DIR / f"{split}.csv")


def load_all() -> dict[str, pd.DataFrame]:
    return {s: load_split(s) for s in ("train", "dev", "test")}


# ---------------------------------------------------------------------------
# Label distribution
# ---------------------------------------------------------------------------

def label_distribution(df: pd.DataFrame, col: str, name_map: dict) -> pd.DataFrame:
    """count + percentage per class for one split."""
    counts = df[col].value_counts().sort_index()
    total = len(df)
    rows = []
    for val, cnt in counts.items():
        rows.append({
            "label_id": int(val),
            "label_name": name_map.get(int(val), str(val)),
            "count": int(cnt),
            "percentage": round(100.0 * cnt / total, 4),
        })
    return pd.DataFrame(rows)


def imbalance_ratio(dist: pd.DataFrame) -> dict:
    """majority/minority stats from a distribution DataFrame."""
    counts = dist["count"]
    maj = int(counts.max())
    mino = int(counts.min())
    maj_name = dist.loc[counts.idxmax(), "label_name"]
    min_name = dist.loc[counts.idxmin(), "label_name"]
    return {
        "majority_class": maj_name,
        "majority_count": maj,
        "minority_class": min_name,
        "minority_count": mino,
        "ratio_majority_to_minority": round(maj / mino, 3) if mino else None,
        "n_classes": int(len(dist)),
    }


# ---------------------------------------------------------------------------
# Text length
# ---------------------------------------------------------------------------

# PhoBERT BPE tokenizer (optional; only if loadable). Loaded lazily.
_PHOBERT_TOK = None


def _get_phobert_tokenizer():
    global _PHOBERT_TOK
    if _PHOBERT_TOK is None:
        try:
            from transformers import AutoTokenizer
            _PHOBERT_TOK = AutoTokenizer.from_pretrained(
                "vinai/phobert-base-v2", add_prefix_space=True
            )
        except Exception:
            _PHOBERT_TOK = False  # mark as unavailable
    return _PHOBERT_TOK or None


def text_length_stats(df: pd.DataFrame, text_col: str = "text_clean",
                      use_phobert: bool = True) -> dict:
    """char / word / token-length stats for one split."""
    texts = df[text_col].astype(str)
    char_len = texts.str.len()
    word_len = texts.str.split().str.len()

    stats = {
        "n_samples": int(len(df)),
        "chars": _describe(char_len),
        "words": _describe(word_len),
    }

    if use_phobert:
        tok = _get_phobert_tokenizer()
        if tok is not None:
            tok_len = texts.map(lambda t: len(tok.tokenize(t)))
            stats["phobert_tokens"] = _describe(tok_len)
            stats["phobert_truncated_gt_32"] = int((tok_len > MAX_LENGTH_USED).sum())
            stats["phobert_truncated_gt_32_pct"] = round(
                100.0 * (tok_len > MAX_LENGTH_USED).sum() / len(df), 4
            )
        else:
            stats["phobert_tokens"] = None
            stats["phobert_note"] = "PhoBERT tokenizer unavailable"
    return stats


def _describe(series: pd.Series) -> dict:
    s = series.astype(float)
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": round(float(s.mean()), 3),
        "median": float(s.median()),
        "p90": round(float(s.quantile(0.90)), 3),
        "p95": round(float(s.quantile(0.95)), 3),
        "p99": round(float(s.quantile(0.99)), 3),
        "std": round(float(s.std()), 3),
    }


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

# Fixed, embedded Vietnamese stopword list (reproducible, no external download).
# Curated for this dataset's domain (student feedback). Lowercase, single tokens.
VIETNAMESE_STOPWORDS = frozenset("""
và của là các cho được có không một những trong để với này đã mà cũng
khi thì như về từ ra nên hay rất nhiều quá hơn lại còn chỉ mỗi đó ở
theo lên xuống bị do phải sẽ đang vừa mới đây kia nọ tôi bạn họ chúng ta
anh chị em ông bà cô thầy nó họ này kia ai gì đâu bao nhiêu sao thế nào
vậy nhé nhỉ ạ ơi nhỉ ha hả chứ đấy nhé thôi luôn cả mọi từng riêng chung
toàn cùng theo trên dưới trong ngoài giữa qua lại sang tới lui đi đến về
rồi chưa đã sẽ đang vẫn cứ mãi luôn từng luôn mãi đều đều hết cả cùng
""".split())

# Tokens shorter than this are dropped (punctuation, stray chars).
MIN_TOKEN_LEN = 2

_TOKEN_RE = re.compile(r"[0-9a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệ"
                       r"ỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+", re.IGNORECASE)


def tokenize_for_keywords(text: str) -> list[str]:
    """Lowercase, keep word tokens, drop stopwords + short/punct tokens."""
    tokens = _TOKEN_RE.findall(str(text).lower())
    return [t for t in tokens
            if len(t) >= MIN_TOKEN_LEN and t not in VIETNAMESE_STOPWORDS]


def keyword_frequency(df: pd.DataFrame, text_col: str = "text_clean",
                      top_n: int = 30) -> pd.DataFrame:
    """Top-N keywords over the whole split."""
    counter = Counter()
    for t in df[text_col]:
        counter.update(tokenize_for_keywords(t))
    rows = [{"keyword": w, "count": c} for w, c in counter.most_common(top_n)]
    return pd.DataFrame(rows)


def keyword_frequency_by(df: pd.DataFrame, group_col: str, name_map: dict,
                         text_col: str = "text_clean", top_n: int = 20) -> dict:
    """Top-N keywords per class -> {class_name: DataFrame}."""
    out = {}
    for val, name in sorted(name_map.items()):
        sub = df[df[group_col] == val]
        out[name] = keyword_frequency(sub, text_col, top_n)
    return out


# ---------------------------------------------------------------------------
# Sentiment x Topic
# ---------------------------------------------------------------------------

def sentiment_topic_crosstab(df: pd.DataFrame, sent_map: dict,
                             topic_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw counts, row-normalised pct) contingency tables."""
    ct = pd.crosstab(df["sentiment"], df["topic"])
    ct.index = [sent_map[i] for i in ct.index]
    ct.columns = [topic_map[i] for i in ct.columns]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100.0
    return ct, pct.round(2)


# ---------------------------------------------------------------------------
# Emoji / special signals
# ---------------------------------------------------------------------------

# Broad emoji / pictograph matcher (Unicode-aware via regex lib).
_EMOJI_RE = None


def _emoji_regex():
    global _EMOJI_RE
    if _EMOJI_RE is None:
        try:
            import regex as reg
            _EMOJI_RE = reg.compile(r"\p{Extended_Pictographic}")
        except Exception:
            import re as _re
            _EMOJI_RE = _re.compile(
                "[\U0001F000-\U0001FAFF☀-➿⬀-⯿\U0001F1E6-\U0001F1FF]"
            )
    return _EMOJI_RE


def emoji_stats(df: pd.DataFrame, text_col: str = "text_clean") -> dict:
    """Emoji prevalence + frequency for one split."""
    er = _emoji_regex()
    texts = df[text_col].astype(str)
    mask = texts.map(lambda t: bool(er.search(t)))
    counter = Counter()
    for t in texts[mask]:
        counter.update(er.findall(t))
    n = int(len(df))
    n_emoji = int(mask.sum())
    return {
        "n_samples": n,
        "n_with_emoji": n_emoji,
        "pct_with_emoji": round(100.0 * n_emoji / n, 4),
        "top_emojis": [{"emoji": e, "count": c}
                       for e, c in counter.most_common(20)],
        "n_distinct_emoji": int(len(counter)),
    }
