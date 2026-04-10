from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.config import (
    MODEL_FAMILIES,
    PipelineConfig,
    binary_dir,
    comparison_dir,
    multilabel_dir,
)
from utils.io_utils import choose_text_column, coerce_binary_labels, load_parquet_frame, save_frame, save_json


def resolve_model_names(requested: list[str] | None = None) -> list[str]:
    allowed = list(MODEL_FAMILIES)
    if not requested:
        return list(allowed)
    names = [name.strip() for name in requested if name.strip()]
    invalid = sorted(set(names) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported model names: {', '.join(invalid)}")
    return names


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of the retained model families to run.",
    )


def load_labeled_data(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    columns = ["id", "comment_text", "comment_text_clean", "comment_text_tfidf"] + config.label_cols
    train = load_parquet_frame(config, "train", columns=columns)
    test = load_parquet_frame(config, "test_labeled", columns=columns)
    train = coerce_binary_labels(train, config.label_cols)
    test = coerce_binary_labels(test, config.label_cols)
    text_column = choose_text_column(train, config.text_priority)
    return train, test, text_column


def approach_output_dir(config: PipelineConfig, model_name: str, mode: str, label_name: str | None = None) -> Path:
    if mode == "multilabel":
        return multilabel_dir(config, model_name)
    if mode == "binary":
        if not label_name:
            raise ValueError("A binary label name is required for binary outputs.")
        return binary_dir(config, model_name, label_name)
    raise ValueError(f"Unsupported mode: {mode}")


def save_metric_bundle(
    root: Path,
    label_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    threshold_frame: pd.DataFrame | None = None,
    logger: object | None = None,
) -> None:
    save_frame(label_metrics, root / "final_metrics.csv", index=False)
    save_frame(aggregate_metrics, root / "aggregate_metrics.csv", index=False)
    if threshold_frame is not None and not threshold_frame.empty:
        save_frame(threshold_frame, root / "threshold_summary.csv", index=False)
    if logger is not None:
        logger.saved(root / "final_metrics.csv", indent=1)
        logger.saved(root / "aggregate_metrics.csv", indent=1)
        if threshold_frame is not None and not threshold_frame.empty:
            logger.saved(root / "threshold_summary.csv", indent=1)


def save_search_bundle(
    root: Path,
    best_params: dict,
    search_frame: pd.DataFrame,
    tfidf_summary: pd.DataFrame,
    cv_result: dict,
    logger: object | None = None,
) -> None:
    save_json(best_params, root / "best_params.json")
    save_json(cv_result["thresholds"], root / "best_thresholds.json")
    save_frame(search_frame, root / "pretuning_results" / "cv_search_results.csv", index=False)
    save_frame(tfidf_summary, root / "pretuning_results" / "tfidf_param_comparison.csv", index=False)
    save_frame(cv_result["fold_metrics"], root / "cv_results.csv", index=False)
    save_frame(cv_result["fold_label_metrics"], root / "cv_label_results.csv", index=False)
    save_frame(cv_result["threshold_frame"], root / "threshold_summary.csv", index=False)
    if logger is not None:
        logger.saved(root / "best_params.json", indent=1)
        logger.saved(root / "best_thresholds.json", indent=1)
        logger.saved(root / "cv_results.csv", indent=1)
        logger.saved(root / "cv_label_results.csv", indent=1)
        logger.saved(root / "threshold_summary.csv", indent=1)
        logger.saved(root / "pretuning_results" / "cv_search_results.csv", indent=1)
        logger.saved(root / "pretuning_results" / "tfidf_param_comparison.csv", indent=1)


def comparison_frame_from_metrics(
    multilabel_metrics: pd.DataFrame,
    binary_metrics: pd.DataFrame,
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    metric_cols = metric_cols or ["precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"]
    merged = multilabel_metrics[["label"] + metric_cols].merge(
        binary_metrics[["label"] + metric_cols],
        on="label",
        suffixes=("_multilabel", "_binary"),
    )
    for metric_col in metric_cols:
        merged[f"{metric_col}_delta_binary_minus_multilabel"] = (
            merged[f"{metric_col}_binary"] - merged[f"{metric_col}_multilabel"]
        )
    return merged


def save_comparison_bundle(
    root: Path,
    comparison_frame: pd.DataFrame,
    logger: object | None = None,
) -> None:
    save_frame(comparison_frame, root / "multilabel_vs_binary_summary.csv", index=False)
    if logger is not None:
        logger.saved(root / "multilabel_vs_binary_summary.csv", indent=1)
