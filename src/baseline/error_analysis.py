"""Error analysis for baseline Linear SVM predictions on TEST set.

Reads existing prediction files from results/baseline/ — does NOT
re-generate predictions, does NOT touch data/raw or data/processed.
Outputs to results/error_analysis/.
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PRED_DIR = BASE_DIR / "results" / "baseline"
OUT_DIR = BASE_DIR / "results" / "error_analysis"

SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
TOPIC_MAP = {0: "Lecturer", 1: "Training_program", 2: "Facility", 3: "Others"}


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"id", "text", "true_label", "predicted_label", "correct"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} thiếu cột: {missing}")
    df["true_label"] = df["true_label"].astype(int)
    df["predicted_label"] = df["predicted_label"].astype(int)
    return df


def error_pairs(errors: pd.DataFrame, total_class_counts: dict) -> pd.DataFrame:
    """Build confusion/error-pair table: count, % of errors, % of true class."""
    total_errors = len(errors)
    grouped = (
        errors.groupby(["true_label_name", "predicted_label_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    grouped["pct_of_errors"] = (grouped["count"] / total_errors * 100).round(2)
    grouped["pct_of_true_class"] = grouped.apply(
        lambda r: round(r["count"] / total_class_counts[r["true_label_name"]] * 100, 2),
        axis=1,
    )
    return grouped


def text_length_stats(errors: pd.DataFrame, all_df: pd.DataFrame) -> dict:
    """Compare text length (words) between errors and correct predictions."""
    correct = all_df[all_df["correct"] == 1]
    err_lens = errors["text"].str.split().str.len()
    cor_lens = correct["text"].str.split().str.len()
    return {
        "error_mean_words": round(float(err_lens.mean()), 2),
        "error_median_words": float(err_lens.median()),
        "correct_mean_words": round(float(cor_lens.mean()), 2),
        "correct_median_words": float(cor_lens.median()),
        "error_short_5w": int((err_lens <= 5).sum()),
        "error_short_5w_pct": round((err_lens <= 5).mean() * 100, 2),
        "correct_short_5w_pct": round((cor_lens <= 5).mean() * 100, 2),
    }


def representative_samples(errors: pd.DataFrame, pair: tuple, n: int = 10) -> pd.DataFrame:
    sub = errors[
        (errors["true_label_name"] == pair[0])
        & (errors["predicted_label_name"] == pair[1])
    ].copy()
    sub["n_words"] = sub["text"].str.split().str.len()
    # Prefer shorter samples (cleaner to display) but keep a spread
    sub = sub.sort_values("n_words")
    return sub.head(n)


def analyze_task(pred_file: str, label_map: dict, task: str) -> dict:
    df = load_predictions(PRED_DIR / pred_file)
    total = len(df)
    errors = df[df["correct"] == 0].copy()
    n_errors = len(errors)

    # Validation: error count must match test - correct
    n_correct = int(df["correct"].sum())
    assert n_correct + n_errors == total, f"{task}: correct+errors != total"
    assert not df["id"].duplicated().any(), f"{task}: duplicate ids"

    true_counts = df["true_label_name"].value_counts().to_dict()
    pairs = error_pairs(errors, true_counts)
    len_stats = text_length_stats(errors, df)

    return {
        "task": task,
        "total": total,
        "n_errors": n_errors,
        "accuracy": round(n_correct / total, 4),
        "true_class_counts": true_counts,
        "error_pairs": pairs,
        "length_stats": len_stats,
        "errors_df": errors,
        "all_df": df,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sent = analyze_task(
        "sentiment_linear_svm_test_predictions.csv", SENTIMENT_MAP, "sentiment"
    )
    topic = analyze_task(
        "topic_linear_svm_test_predictions.csv", TOPIC_MAP, "topic"
    )

    # ---- Error-pair tables ----
    sent["error_pairs"].to_csv(OUT_DIR / "sentiment_error_summary.csv", index=False)
    topic["error_pairs"].to_csv(OUT_DIR / "topic_error_summary.csv", index=False)

    # ---- Representative samples: top-3 error pairs per task ----
    sent_samples = []
    for pair in sent["error_pairs"].head(3).itertuples():
        sent_samples.append(
            representative_samples(
                sent["errors_df"],
                (pair.true_label_name, pair.predicted_label_name),
            )
        )
    sent_samples_df = pd.concat(sent_samples)[
        ["id", "text", "true_label_name", "predicted_label_name"]
    ]
    sent_samples_df.columns = ["id", "text", "true_label", "predicted_label"]
    sent_samples_df.to_csv(OUT_DIR / "sentiment_error_samples.csv", index=False)

    topic_samples = []
    for pair in topic["error_pairs"].head(4).itertuples():
        topic_samples.append(
            representative_samples(
                topic["errors_df"],
                (pair.true_label_name, pair.predicted_label_name),
            )
        )
    topic_samples_df = pd.concat(topic_samples)[
        ["id", "text", "true_label_name", "predicted_label_name"]
    ]
    topic_samples_df.columns = ["id", "text", "true_label", "predicted_label"]
    topic_samples_df.to_csv(OUT_DIR / "topic_error_samples.csv", index=False)

    # ---- Statistics JSON ----
    stats = {
        "sentiment": {
            "total_test": sent["total"],
            "n_errors": sent["n_errors"],
            "accuracy": sent["accuracy"],
            "error_pairs": sent["error_pairs"].to_dict(orient="records"),
            "length_stats": sent["length_stats"],
            "true_class_counts": sent["true_class_counts"],
        },
        "topic": {
            "total_test": topic["total"],
            "n_errors": topic["n_errors"],
            "accuracy": topic["accuracy"],
            "error_pairs": topic["error_pairs"].to_dict(orient="records"),
            "length_stats": topic["length_stats"],
            "true_class_counts": topic["true_class_counts"],
        },
    }
    with open(OUT_DIR / "error_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---- Console summary ----
    for res in (sent, topic):
        print(f"\n{'='*60}")
        print(f"{res['task'].upper()} — errors: {res['n_errors']}/{res['total']} "
              f"(acc={res['accuracy']})")
        print(res["error_pairs"].to_string(index=False))
        print(f"Length stats: {res['length_stats']}")

    print(f"\nArtifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
