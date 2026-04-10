from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src.features.lexicons import IDENTITY_GROUPS
from src.preprocessing.clean_text import build_length_buckets
from src.utils.config import PipelineConfig
from src.utils.io_helpers import write_markdown


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _top_ngrams(texts: pd.Series, top_n: int = 20) -> pd.DataFrame:
    if texts.empty:
        return pd.DataFrame(columns=["ngram", "count"])

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["ngram", "count"])

    counts = np.asarray(matrix.sum(axis=0)).ravel()
    order = counts.argsort()[::-1][:top_n]
    features = vectorizer.get_feature_names_out()
    return pd.DataFrame({"ngram": features[order], "count": counts[order]})


def run_eda(train_df: pd.DataFrame, figures_dir: Path, output_dir: Path, config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    sns.set_theme(style="whitegrid")

    label_counts = train_df[config.labels].sum().sort_values(ascending=False).rename("positive_count").reset_index()
    label_counts.columns = ["label", "positive_count"]
    label_counts["prevalence"] = label_counts["positive_count"] / len(train_df)
    label_counts.to_csv(output_dir / "label_prevalence.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=label_counts, x="label", y="positive_count")
    plt.xticks(rotation=30, ha="right")
    plt.title("Label Prevalence")
    _save_plot(figures_dir / "label_prevalence.png")

    cooccurrence = train_df[config.labels].T @ train_df[config.labels]
    cooccurrence.to_csv(output_dir / "label_cooccurrence.csv")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cooccurrence, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Label Co-occurrence")
    _save_plot(figures_dir / "label_cooccurrence_heatmap.png")

    active_count = train_df[config.labels].sum(axis=1).rename("active_labels")
    active_count.value_counts().sort_index().rename_axis("active_labels").reset_index(name="count").to_csv(
        output_dir / "active_label_count_distribution.csv",
        index=False,
    )
    plt.figure(figsize=(7, 4))
    sns.countplot(x=active_count)
    plt.title("Active Label Count Distribution")
    _save_plot(figures_dir / "active_label_count_distribution.png")

    length_df = pd.DataFrame(
        {
            "char_count": train_df["comment_text_clean"].str.len(),
            "word_count": train_df["comment_text_clean"].str.split().str.len(),
        }
    )
    length_df.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).transpose().to_csv(output_dir / "comment_length_summary.csv")

    plt.figure(figsize=(8, 4))
    sns.histplot(length_df["char_count"], bins=60)
    plt.title("Comment Length Distribution (Characters)")
    _save_plot(figures_dir / "char_length_distribution.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(length_df["word_count"], bins=60)
    plt.title("Comment Length Distribution (Words)")
    _save_plot(figures_dir / "word_length_distribution.png")

    length_bucket_df = pd.DataFrame(
        {
            "length_bucket": build_length_buckets(length_df["char_count"]),
            "any_toxic": (train_df[config.labels].sum(axis=1) > 0).astype(int),
        }
    )
    toxicity_by_bucket = length_bucket_df.groupby("length_bucket", observed=False)["any_toxic"].agg(["mean", "count"]).reset_index()
    toxicity_by_bucket.to_csv(output_dir / "toxicity_rate_by_length_bucket.csv", index=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=toxicity_by_bucket, x="length_bucket", y="mean")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Any Toxic Rate")
    plt.title("Toxicity Rate by Length Bucket")
    _save_plot(figures_dir / "toxicity_by_length_bucket.png")

    top_overall = _top_ngrams(train_df["comment_text_tfidf"], top_n=25)
    top_overall.to_csv(output_dir / "top_ngrams_overall.csv", index=False)
    for label in config.labels:
        subset = train_df.loc[train_df[label] == 1, "comment_text_tfidf"]
        _top_ngrams(subset, top_n=20).to_csv(output_dir / f"top_ngrams_{label}.csv", index=False)

    representative_rows: List[pd.DataFrame] = []
    for label in config.labels:
        subset = train_df.loc[train_df[label] == 1, ["id", "comment_text_raw"]].head(5).copy()
        subset.insert(1, "label", label)
        representative_rows.append(subset)
    pd.concat(representative_rows, ignore_index=True).to_csv(output_dir / "representative_examples.csv", index=False)

    duplicate_counts = train_df["comment_text_raw"].value_counts()
    duplicate_report = pd.DataFrame(
        {
            "exact_duplicate_rows": [int((duplicate_counts > 1).sum())],
            "max_duplicate_frequency": [int(duplicate_counts.max())],
        }
    )
    duplicate_report.to_csv(output_dir / "duplicate_comment_summary.csv", index=False)

    subgroup_records = []
    text_series = train_df["comment_text_tfidf"]
    for group_name, terms in IDENTITY_GROUPS.items():
        pattern = r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b"
        subgroup_records.append({"subgroup": group_name, "matched_rows": int(text_series.str.contains(pattern, regex=True).sum())})
    pd.DataFrame(subgroup_records).to_csv(output_dir / "subgroup_coverage_counts.csv", index=False)

    summary = f"""
# Data Exploration Summary

- Train rows: {len(train_df):,}
- Labels analyzed: {", ".join(config.labels)}
- Maximum observed character length: {int(length_df["char_count"].max()):,}
- Median character length: {int(length_df["char_count"].median()):,}
- Multi-label rows: {int((train_df[config.labels].sum(axis=1) > 1).sum()):,}
- Exact duplicate comment texts: {int((duplicate_counts > 1).sum()):,}
- We treat subgroup coverage as a proxy identity-term slice, not as direct group membership.
"""
    write_markdown(summary, output_dir / "eda_summary.md")
    return {"label_prevalence": label_counts, "cooccurrence": cooccurrence, "length_summary": length_df.describe().transpose()}
