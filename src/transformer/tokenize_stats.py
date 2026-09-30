"""Tokenization analysis — run BEFORE training to pick max_length."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer

from src.transformer.config import (
    DATA_DIR,
    MODEL_NAME,
    RESULTS_DIR,
)
from src.transformer.dataset import load_split, token_length_stats


def main():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    stats = {}
    for split in ("train", "dev", "test"):
        df = load_split(DATA_DIR / f"{split}.csv")
        s = token_length_stats(df, tokenizer, text_col="text_clean")
        stats[split] = s
        print(f"\n=== {split} ({s['count']} samples) ===")
        print(f"  min={s['min']}  median={s['median']}  mean={s['mean']}")
        print(f"  p90={s['p90']}  p95={s['p95']}  p99={s['p99']}  max={s['max']}")
        print(f"  >64: {s['truncated_at_64_pct']}%  "
              f">96: {s['truncated_at_96_pct']}%  "
              f">128: {s['truncated_at_128_pct']}%")

    out = RESULTS_DIR / "token_length_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
