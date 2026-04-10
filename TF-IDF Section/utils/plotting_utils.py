from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from utils.io_utils import ensure_dir


def _finish(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bar(frame: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(frame[x_col], frame[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    _finish(path)


def plot_histogram(values: pd.Series | np.ndarray, title: str, xlabel: str, path: Path, bins: int = 50) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    _finish(path)


def plot_model_metric_comparison(frame: pd.DataFrame, metric_col: str, title: str, path: Path) -> None:
    plot_frame = frame.sort_values(metric_col, ascending=False).copy()
    plt.figure(figsize=(9, 4.8))
    plt.bar(plot_frame["model_name"], plot_frame[metric_col])
    plt.title(title)
    plt.ylabel(metric_col)
    plt.xticks(rotation=35, ha="right")
    _finish(path)


def plot_threshold_curve(frame: pd.DataFrame, label_name: str, path: Path) -> None:
    plot_frame = frame[frame["label"] == label_name].copy()
    if plot_frame.empty:
        return
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(plot_frame["threshold"], plot_frame["f1"], label="F1")
    plt.plot(plot_frame["threshold"], plot_frame["precision"], label="Precision")
    plt.plot(plot_frame["threshold"], plot_frame["recall"], label="Recall")
    plt.title(f"Threshold diagnostics: {label_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    _finish(path)


def plot_curve_grid(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    label_cols: list[str],
    path: Path,
    curve_type: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for index, label_name in enumerate(label_cols):
        axis = axes[index]
        label_true = y_true[:, index]
        label_score = y_scores[:, index]
        if len(np.unique(label_true)) < 2:
            axis.axis("off")
            axis.set_title(f"{label_name}: insufficient variation")
            continue
        if curve_type == "pr":
            precision, recall, _ = precision_recall_curve(label_true, label_score)
            axis.plot(recall, precision)
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            axis.set_title(f"PR: {label_name}")
        else:
            fpr, tpr, _ = roc_curve(label_true, label_score)
            axis.plot(fpr, tpr)
            axis.plot([0, 1], [0, 1], linestyle="--")
            axis.set_xlabel("FPR")
            axis.set_ylabel("TPR")
            axis.set_title(f"ROC: {label_name}")
    _finish(path)


def plot_top_features(frame: pd.DataFrame, label_name: str, direction: str, path: Path) -> None:
    plot_frame = frame[(frame["label"] == label_name) & (frame["direction"] == direction)].copy()
    if plot_frame.empty:
        return
    ascending = direction == "negative"
    plot_frame = plot_frame.sort_values("weight", ascending=ascending)
    plt.figure(figsize=(9, 5))
    plt.barh(plot_frame["feature"], plot_frame["weight"])
    plt.title(f"Top {direction} features: {label_name}")
    plt.xlabel("Weight")
    _finish(path)
