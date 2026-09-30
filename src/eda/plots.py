"""Plotting helpers for EDA — matplotlib only (Agg backend, no display).

Every function saves a PNG and returns the saved Path. Nothing here modifies data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

_SENT_COLORS = {"Negative": "#d62728", "Neutral": "#7f7f7f", "Positive": "#2ca02c"}
_TOPIC_COLORS = {
    "Lecturer": "#1f77b4",
    "Training_program": "#ff7f0e",
    "Facility": "#2ca02c",
    "Others": "#9467bd",
}


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def grouped_label_bars(dists: dict[str, pd.DataFrame], name_map: dict,
                       title: str, out: Path) -> Path:
    """Grouped bar chart of class counts across splits.

    dists: {split: DataFrame[label_name, count]}
    """
    splits = list(dists.keys())
    labels = [name_map[i] for i in sorted(name_map)]
    x = np.arange(len(labels))
    width = 0.8 / len(splits)

    fig, ax = plt.subplots(figsize=(8, 5))
    for si, split in enumerate(splits):
        d = dists[split].set_index("label_name").reindex(labels)
        ax.bar(x + si * width - 0.4 + width / 2, d["count"].fillna(0),
               width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(title="split")
    return _save(fig, out)


def label_pct_bars(dists: dict[str, pd.DataFrame], name_map: dict,
                   title: str, out: Path) -> Path:
    """Grouped bar of class percentage across splits."""
    splits = list(dists.keys())
    labels = [name_map[i] for i in sorted(name_map)]
    x = np.arange(len(labels))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(8, 5))
    for si, split in enumerate(splits):
        d = dists[split].set_index("label_name").reindex(labels)
        ax.bar(x + si * width - 0.4 + width / 2, d["percentage"].fillna(0),
               width, label=split)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("%"); ax.set_title(title); ax.legend(title="split")
    return _save(fig, out)


def length_histogram(dfs: dict[str, pd.Series], title: str, out: Path,
                     bins: int = 60) -> Path:
    """Overlaid histogram of text length (words) per split."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for split, series in dfs.items():
        ax.hist(series, bins=bins, alpha=0.45, label=f"{split}", density=True)
    ax.set_xlabel("words per comment")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    return _save(fig, out)


def length_boxplot(dfs: dict[str, pd.Series], title: str, out: Path) -> Path:
    """Boxplot of text length per split."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([dfs[s].values for s in dfs], labels=list(dfs.keys()),
               showfliers=False)
    ax.set_ylabel("words per comment")
    ax.set_title(title)
    return _save(fig, out)


def top_keywords_bar(kw_df: pd.DataFrame, title: str, out: Path,
                     top_n: int = 25) -> Path:
    """Horizontal bar of top keywords."""
    d = kw_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(d))))
    ax.barh(d["keyword"], d["count"], color="#1f77b4")
    ax.set_xlabel("count")
    ax.set_title(title)
    return _save(fig, out)


def top_keywords_grid(kw_by: dict[str, pd.DataFrame], title: str, out: Path,
                      top_n: int = 15) -> Path:
    """Grid of per-class top-keyword horizontal bars."""
    names = list(kw_by.keys())
    ncol = 2
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        d = kw_by[name].head(top_n).iloc[::-1]
        color = _SENT_COLORS.get(name) or _TOPIC_COLORS.get(name) or "#1f77b4"
        ax.barh(d["keyword"], d["count"], color=color)
        ax.set_title(name)
        ax.set_xlabel("count")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(title)
    return _save(fig, out)


def crosstab_heatmap(counts: pd.DataFrame, title: str, out: Path) -> Path:
    """Annotated heatmap of a contingency table (raw counts).

    Cell colour is log-scaled so that minority cells stay visible next to the
    dominant Positive×Lecturer cell; the annotation still shows raw counts.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    log_vals = np.log10(counts.values + 1)
    im = ax.imshow(log_vals, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(counts.columns)))
    ax.set_xticklabels(counts.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(counts.index)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("topic"); ax.set_ylabel("sentiment")
    vmax = log_vals.max()
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = int(counts.values[i, j])
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if log_vals[i, j] > 0.55 * vmax else "black",
                    fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(count+1)")
    return _save(fig, out)


def emoji_bar(top_emojis: list[dict], title: str, out: Path, top_n: int = 20) -> Path:
    """Bar chart of top emojis by count.

    The default font has no emoji glyphs, so each bar is labelled with the
    emoji's Unicode name (unicodedata) beneath the codepoint, which renders
    reliably, and the emoji itself is shown if the font supports it.
    """
    import unicodedata
    d = top_emojis[:top_n]
    vals = [e["count"] for e in d]

    def label_for(e: str) -> str:
        try:
            name = unicodedata.name(e).title().replace("Face", "").strip()
        except Exception:
            name = f"U+{ord(e):X}"
        return f"{e}\n{name}"

    labels = [label_for(e["emoji"]) for e in d]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(range(len(vals)), vals, color="#9467bd")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("count")
    ax.set_title(title)
    return _save(fig, out)
