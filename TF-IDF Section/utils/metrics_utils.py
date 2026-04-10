from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return np.nan


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def default_threshold_value(score_kind: str) -> float:
    return 0.5 if score_kind == "probability" else 0.0


def default_thresholds(label_cols: list[str], score_kind: str) -> dict[str, float]:
    value = default_threshold_value(score_kind)
    return {label: value for label in label_cols}


def threshold_grid(scores: np.ndarray, score_kind: str) -> np.ndarray:
    if score_kind == "probability":
        return np.linspace(0.05, 0.95, 91)
    quantiles = np.linspace(0.01, 0.99, 99)
    grid = np.unique(np.quantile(scores, quantiles))
    return np.unique(np.concatenate(([0.0], grid)))


def apply_thresholds(score_matrix: np.ndarray, thresholds: dict[str, float], label_cols: list[str]) -> pd.DataFrame:
    data = {
        label: (score_matrix[:, idx] >= thresholds[label]).astype(int)
        for idx, label in enumerate(label_cols)
    }
    return pd.DataFrame(data)


def tune_thresholds(
    y_true: pd.DataFrame,
    score_matrix: np.ndarray,
    label_cols: list[str],
    score_kind: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    best_thresholds = default_thresholds(label_cols, score_kind)
    rows: list[dict[str, object]] = []
    default_value = default_threshold_value(score_kind)

    for idx, label in enumerate(label_cols):
        label_true = y_true[label].to_numpy()
        label_scores = score_matrix[:, idx]
        best_f1 = -1.0
        best_threshold = default_value
        for threshold in threshold_grid(label_scores, score_kind):
            label_pred = (label_scores >= threshold).astype(int)
            precision = precision_score(label_true, label_pred, zero_division=0)
            recall = recall_score(label_true, label_pred, zero_division=0)
            f1 = f1_score(label_true, label_pred, zero_division=0)
            rows.append(
                {
                    "label": label,
                    "threshold": float(threshold),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "predicted_positive_rate": float(label_pred.mean()),
                }
            )
            better = f1 > best_f1
            tie_break = np.isclose(f1, best_f1) and abs(float(threshold) - default_value) < abs(best_threshold - default_value)
            if better or tie_break:
                best_f1 = f1
                best_threshold = float(threshold)
        best_thresholds[label] = best_threshold

    return best_thresholds, pd.DataFrame(rows)


def per_label_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    score_matrix: np.ndarray,
    label_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(label_cols):
        label_true = y_true[label].to_numpy()
        label_pred = y_pred[label].to_numpy()
        label_scores = score_matrix[:, idx]
        tp = int(((label_true == 1) & (label_pred == 1)).sum())
        tn = int(((label_true == 0) & (label_pred == 0)).sum())
        fp = int(((label_true == 0) & (label_pred == 1)).sum())
        fn = int(((label_true == 1) & (label_pred == 0)).sum())
        rows.append(
            {
                "label": label,
                "support": int(label_true.sum()),
                "prevalence": float(label_true.mean()),
                "predicted_positive_rate": float(label_pred.mean()),
                "precision": precision_score(label_true, label_pred, zero_division=0),
                "recall": recall_score(label_true, label_pred, zero_division=0),
                "f1": f1_score(label_true, label_pred, zero_division=0),
                "accuracy": accuracy_score(label_true, label_pred),
                "roc_auc": _safe_metric(roc_auc_score, label_true, label_scores),
                "pr_auc": _safe_metric(average_precision_score, label_true, label_scores),
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "false_positive_rate": _safe_divide(fp, fp + tn),
                "false_negative_rate": _safe_divide(fn, fn + tp),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    score_matrix: np.ndarray,
    label_cols: list[str],
) -> pd.DataFrame:
    true_matrix = y_true[label_cols].to_numpy()
    pred_matrix = y_pred[label_cols].to_numpy()

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_matrix,
        pred_matrix,
        average="macro",
        zero_division=0,
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        true_matrix,
        pred_matrix,
        average="micro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_matrix,
        pred_matrix,
        average="weighted",
        zero_division=0,
    )

    label_frame = per_label_metrics(y_true, y_pred, score_matrix, label_cols)
    rows = [
        {"metric": "macro_precision", "value": float(macro_precision)},
        {"metric": "macro_recall", "value": float(macro_recall)},
        {"metric": "macro_f1", "value": float(macro_f1)},
        {"metric": "micro_precision", "value": float(micro_precision)},
        {"metric": "micro_recall", "value": float(micro_recall)},
        {"metric": "micro_f1", "value": float(micro_f1)},
        {"metric": "weighted_precision", "value": float(weighted_precision)},
        {"metric": "weighted_recall", "value": float(weighted_recall)},
        {"metric": "weighted_f1", "value": float(weighted_f1)},
        {"metric": "subset_accuracy", "value": accuracy_score(true_matrix, pred_matrix)},
        {"metric": "hamming_loss", "value": hamming_loss(true_matrix, pred_matrix)},
        {"metric": "jaccard_micro", "value": jaccard_score(true_matrix, pred_matrix, average="micro", zero_division=0)},
        {"metric": "macro_pr_auc", "value": float(label_frame["pr_auc"].mean(skipna=True))},
        {"metric": "macro_roc_auc", "value": float(label_frame["roc_auc"].mean(skipna=True))},
        {"metric": "micro_pr_auc", "value": _safe_metric(average_precision_score, true_matrix, score_matrix, average="micro")},
        {"metric": "micro_roc_auc", "value": _safe_metric(roc_auc_score, true_matrix, score_matrix, average="micro")},
    ]
    return pd.DataFrame(rows)


def metric_lookup(frame: pd.DataFrame) -> dict[str, float]:
    return {row.metric: float(row.value) for row in frame.itertuples(index=False)}


def metric_value(frame: pd.DataFrame, metric_name: str) -> float:
    return metric_lookup(frame).get(metric_name, np.nan)


def selection_values(frame: pd.DataFrame, metric_names: list[str]) -> tuple[float, ...]:
    lookup = metric_lookup(frame)
    return tuple(float(lookup.get(metric_name, np.nan)) for metric_name in metric_names)


def ranking_score(frame: pd.DataFrame, weights: dict[str, float]) -> float:
    lookup = metric_lookup(frame)
    total = 0.0
    for metric_name, weight in weights.items():
        metric_value = lookup.get(metric_name, np.nan)
        if np.isnan(metric_value):
            continue
        total += weight * metric_value
    return float(total)


def error_analysis_samples(
    source_frame: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    score_matrix: np.ndarray,
    thresholds: dict[str, float],
    label_cols: list[str],
    text_col: str,
    top_n: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(label_cols):
        label_true = y_true[label].to_numpy()
        label_pred = y_pred[label].to_numpy()
        label_scores = score_matrix[:, idx]
        threshold = thresholds[label]
        fp_idx = np.where((label_true == 0) & (label_pred == 1))[0]
        fn_idx = np.where((label_true == 1) & (label_pred == 0))[0]
        fp_sorted = fp_idx[np.argsort(-(label_scores[fp_idx] - threshold))]
        fn_sorted = fn_idx[np.argsort(-(threshold - label_scores[fn_idx]))]
        for error_type, selected in [("false_positive", fp_sorted), ("false_negative", fn_sorted)]:
            for row_idx in selected[:top_n]:
                rows.append(
                    {
                        "label": label,
                        "error_type": error_type,
                        "id": source_frame.iloc[row_idx]["id"],
                        "text": source_frame.iloc[row_idx][text_col],
                        "score": float(label_scores[row_idx]),
                        "threshold": float(threshold),
                        "margin_from_threshold": float(label_scores[row_idx] - threshold),
                        "true_label": int(label_true[row_idx]),
                        "predicted_label": int(label_pred[row_idx]),
                    }
                )
    return pd.DataFrame(rows)
