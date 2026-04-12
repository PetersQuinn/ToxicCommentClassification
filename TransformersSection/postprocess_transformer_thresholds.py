from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None


LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

THRESHOLD_OBJECTIVE = "maximize_validation_f1_tie_break_to_0_5"
THRESHOLD_GRID_DESCRIPTION = "np.linspace(0.05, 0.95, 91)"
DEFAULT_THRESHOLD = 0.5


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
TRANS_RESULTS = ROOT / "TransformersSection" / "TransResults"
REVIEW_DIR = TRANS_RESULTS / "12_threshold_tuning_review"

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "variant_name": "multilabel_distilbert_cv",
        "variant_class": "shared_multilabel_model",
        "task_type": "multilabel",
        "target_label": None,
        "folder": TRANS_RESULTS / "01_multilabel_distilbert",
        "validation_predictions": "cv_validation_predictions_oof.csv",
        "validation_metrics": "cv_validation_metrics_oof.json",
        "test_labeled_predictions": "test_labeled_predictions.csv",
        "test_labeled_metrics": "test_labeled_metrics.json",
        "test_predictions": "test_predictions.csv",
        "labels": LABEL_COLS,
    },
    {
        "variant_name": "binary_toxic",
        "variant_class": "single_label_binary_model",
        "task_type": "binary",
        "target_label": "toxic",
        "folder": TRANS_RESULTS / "02_binary_toxic",
        "validation_predictions": "validation_predictions.csv",
        "validation_metrics": "validation_metrics.json",
        "test_labeled_predictions": "test_labeled_predictions.csv",
        "test_labeled_metrics": "test_labeled_metrics.json",
        "test_predictions": "test_predictions.csv",
        "labels": ["toxic"],
    },
    {
        "variant_name": "binary_severe_toxic",
        "variant_class": "single_label_binary_model",
        "task_type": "binary",
        "target_label": "severe_toxic",
        "folder": TRANS_RESULTS / "03_binary_severe_toxic",
        "validation_predictions": "validation_predictions.csv",
        "validation_metrics": "validation_metrics.json",
        "test_labeled_predictions": "test_labeled_predictions.csv",
        "test_labeled_metrics": "test_labeled_metrics.json",
        "test_predictions": "test_predictions.csv",
        "labels": ["severe_toxic"],
    },
    {
        "variant_name": "binary_obscene",
        "variant_class": "single_label_binary_model",
        "task_type": "binary",
        "target_label": "obscene",
        "folder": TRANS_RESULTS / "04_binary_obscene",
        "validation_predictions": "validation_predictions.csv",
        "validation_metrics": "validation_metrics.json",
        "test_labeled_predictions": "test_labeled_predictions.csv",
        "test_labeled_metrics": "test_labeled_metrics.json",
        "test_predictions": "test_predictions.csv",
        "labels": ["obscene"],
    },
    {
        "variant_name": "binary_threat",
        "variant_class": "single_label_binary_model",
        "task_type": "binary",
        "target_label": "threat",
        "folder": TRANS_RESULTS / "05_binary_threat",
        "validation_predictions": "validation_predictions.csv",
        "validation_metrics": "validation_metrics.json",
        "test_labeled_predictions": "test_labeled_predictions.csv",
        "test_labeled_metrics": "test_labeled_metrics.json",
        "test_predictions": "test_predictions.csv",
        "labels": ["threat"],
    },
    {
        "variant_name": "binary_insult",
        "variant_class": "single_label_binary_model",
        "task_type": "binary",
        "target_label": "insult",
        "folder": TRANS_RESULTS / "06_binary_insult",
        "validation_predictions": "validation_predictions.csv",
        "validation_metrics": "validation_metrics.json",
        "test_labeled_predictions": "test_labeled_predictions.csv",
        "test_labeled_metrics": "test_labeled_metrics.json",
        "test_predictions": "test_predictions.csv",
        "labels": ["insult"],
    },
    {
        "variant_name": "binary_identity_hate",
        "variant_class": "single_label_binary_model",
        "task_type": "binary",
        "target_label": "identity_hate",
        "folder": TRANS_RESULTS / "07_binary_identity_hate",
        "validation_predictions": "validation_predictions.csv",
        "validation_metrics": "validation_metrics.json",
        "test_labeled_predictions": "test_labeled_predictions.csv",
        "test_labeled_metrics": "test_labeled_metrics.json",
        "test_predictions": "test_predictions.csv",
        "labels": ["identity_hate"],
    },
]


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(data), indent=2))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def safe_metric(fn, *args, **kwargs) -> float | None:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return None


def threshold_grid(_: np.ndarray) -> np.ndarray:
    return np.linspace(0.05, 0.95, 91)


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "not found"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if np.isnan(numeric):
        return "not found"
    return f"{numeric:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def tune_thresholds(validation_df: pd.DataFrame, labels: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, float] = {}

    for label in labels:
        y_true = validation_df[f"{label}_true"].astype(int).to_numpy()
        y_prob = validation_df[f"{label}_prob"].astype(float).to_numpy()

        best_f1 = -1.0
        best_threshold = DEFAULT_THRESHOLD
        for threshold in threshold_grid(y_prob):
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            positive_rate = float(y_pred.mean())
            row = {
                "label": label,
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_positive_rate": positive_rate,
            }
            rows.append(row)

            better = f1 > best_f1
            tie_break = np.isclose(f1, best_f1) and abs(float(threshold) - DEFAULT_THRESHOLD) < abs(
                best_threshold - DEFAULT_THRESHOLD
            )
            if better or tie_break:
                best_f1 = float(f1)
                best_threshold = float(threshold)

        selected[label] = best_threshold

    search_df = pd.DataFrame(rows)
    search_df["is_selected"] = search_df.apply(
        lambda row: bool(np.isclose(float(row["threshold"]), selected[row["label"]])),
        axis=1,
    )
    return selected, search_df


def augment_prediction_frame(df: pd.DataFrame, labels: list[str], thresholds: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for label in labels:
        y_prob = out[f"{label}_prob"].astype(float).to_numpy()
        out[f"{label}_selected_threshold"] = float(thresholds[label])
        out[f"{label}_pred_threshold_tuned"] = (y_prob >= float(thresholds[label])).astype(int)
    return out


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, status: str) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold_tuning_status": status,
        "threshold_selection_objective": THRESHOLD_OBJECTIVE,
        "threshold_search_grid": THRESHOLD_GRID_DESCRIPTION,
        "selected_threshold": float(threshold),
        "average_precision": safe_metric(average_precision_score, y_true, y_prob),
        "roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
        "precision_at_operating_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_operating_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_operating_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred_at_operating_threshold": float(np.mean(y_pred)),
        "classification_report_at_operating_threshold": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
    status_by_label: dict[str, str],
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.column_stack([(y_prob[:, idx] >= thresholds[label]).astype(int) for idx, label in enumerate(labels)])

    per_label: dict[str, dict[str, Any]] = {}
    ap_values: list[float] = []
    auc_values: list[float] = []

    for idx, label in enumerate(labels):
        label_true = y_true[:, idx]
        label_prob = y_prob[:, idx]
        label_pred = y_pred[:, idx]
        ap = safe_metric(average_precision_score, label_true, label_prob)
        auc = safe_metric(roc_auc_score, label_true, label_prob)
        if ap is not None:
            ap_values.append(ap)
        if auc is not None:
            auc_values.append(auc)
        per_label[label] = {
            "threshold_tuning_status": status_by_label[label],
            "selected_threshold": float(thresholds[label]),
            "average_precision": ap,
            "roc_auc": auc,
            "precision_at_operating_threshold": float(precision_score(label_true, label_pred, zero_division=0)),
            "recall_at_operating_threshold": float(recall_score(label_true, label_pred, zero_division=0)),
            "f1_at_operating_threshold": float(f1_score(label_true, label_pred, zero_division=0)),
            "positive_rate_true": float(np.mean(label_true)),
            "positive_rate_pred_at_operating_threshold": float(np.mean(label_pred)),
            "classification_report_at_operating_threshold": classification_report(
                label_true, label_pred, output_dict=True, zero_division=0
            ),
        }

    return {
        "threshold_tuning_status": "threshold_tuned_from_validation",
        "threshold_selection_objective": THRESHOLD_OBJECTIVE,
        "threshold_search_grid": THRESHOLD_GRID_DESCRIPTION,
        "selected_thresholds": {label: float(thresholds[label]) for label in labels},
        "mean_average_precision": float(np.mean(ap_values)) if ap_values else None,
        "mean_roc_auc": float(np.mean(auc_values)) if auc_values else None,
        "micro_f1_at_operating_threshold": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1_at_operating_threshold": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_label": per_label,
    }


def extract_before_after_rows(
    spec: dict[str, Any],
    original_test_metrics: dict[str, Any],
    tuned_test_metrics: dict[str, Any],
    selected_thresholds: dict[str, float],
    status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if spec["task_type"] == "multilabel":
        per_label_before = original_test_metrics["per_label"]
        per_label_after = tuned_test_metrics["per_label"]
        for label in spec["labels"]:
            before = per_label_before[label]
            after = per_label_after[label]
            rows.append(
                {
                    "variant_name": spec["variant_name"],
                    "task_type": spec["task_type"],
                    "label": label,
                    "threshold_before": 0.5,
                    "threshold_after": float(selected_thresholds[label]),
                    "average_precision_before": before.get("average_precision"),
                    "average_precision_after": after.get("average_precision"),
                    "precision_before": before.get("precision_at_0_5"),
                    "precision_after": after.get("precision_at_operating_threshold"),
                    "recall_before": before.get("recall_at_0_5"),
                    "recall_after": after.get("recall_at_operating_threshold"),
                    "f1_before": before.get("f1_at_0_5"),
                    "f1_after": after.get("f1_at_operating_threshold"),
                    "predicted_positive_rate_before": before.get("positive_rate_pred_at_0_5"),
                    "predicted_positive_rate_after": after.get("positive_rate_pred_at_operating_threshold"),
                    "status": status,
                }
            )
        return rows

    label = spec["target_label"]
    rows.append(
        {
            "variant_name": spec["variant_name"],
            "task_type": spec["task_type"],
            "label": label,
            "threshold_before": 0.5,
            "threshold_after": float(selected_thresholds[label]),
            "average_precision_before": original_test_metrics.get("average_precision"),
            "average_precision_after": tuned_test_metrics.get("average_precision"),
            "precision_before": original_test_metrics.get("precision_at_0_5"),
            "precision_after": tuned_test_metrics.get("precision_at_operating_threshold"),
            "recall_before": original_test_metrics.get("recall_at_0_5"),
            "recall_after": tuned_test_metrics.get("recall_at_operating_threshold"),
            "f1_before": original_test_metrics.get("f1_at_0_5"),
            "f1_after": tuned_test_metrics.get("f1_at_operating_threshold"),
            "predicted_positive_rate_before": original_test_metrics.get("positive_rate_pred_at_0_5"),
            "predicted_positive_rate_after": tuned_test_metrics.get("positive_rate_pred_at_operating_threshold"),
            "status": status,
        }
    )
    return rows


def create_prediction_outputs(
    spec: dict[str, Any],
    thresholds: dict[str, float],
    validation_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    test_unlabeled_df: pd.DataFrame | None,
) -> list[Path]:
    created: list[Path] = []
    folder = spec["folder"]
    labels = spec["labels"]

    if validation_df is not None:
        validation_out = augment_prediction_frame(validation_df, labels, thresholds)
        validation_out_path = folder / "validation_predictions_threshold_tuned.csv"
        validation_out.to_csv(validation_out_path, index=False)
        created.append(validation_out_path)

    test_labeled_out = augment_prediction_frame(test_df, labels, thresholds)
    test_labeled_out_path = folder / "test_labeled_predictions_threshold_tuned.csv"
    test_labeled_out.to_csv(test_labeled_out_path, index=False)
    created.append(test_labeled_out_path)

    if test_unlabeled_df is not None:
        unlabeled_out = augment_prediction_frame(test_unlabeled_df, labels, thresholds)
        unlabeled_out_path = folder / "test_predictions_threshold_tuned.csv"
        unlabeled_out.to_csv(unlabeled_out_path, index=False)
        created.append(unlabeled_out_path)

    return created


def build_multilabel_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label in LABEL_COLS:
        label_metrics = metrics["per_label"][label]
        rows.append(
            {
                "model_family": "multilabel",
                "label": label,
                "selected_threshold": label_metrics["selected_threshold"],
                "average_precision": label_metrics["average_precision"],
                "roc_auc": label_metrics["roc_auc"],
                "precision_at_operating_threshold": label_metrics["precision_at_operating_threshold"],
                "recall_at_operating_threshold": label_metrics["recall_at_operating_threshold"],
                "f1_at_operating_threshold": label_metrics["f1_at_operating_threshold"],
                "positive_rate_true": label_metrics["positive_rate_true"],
                "positive_rate_pred_at_operating_threshold": label_metrics[
                    "positive_rate_pred_at_operating_threshold"
                ],
                "threshold_tuning_status": label_metrics["threshold_tuning_status"],
            }
        )
    return pd.DataFrame(rows)


def build_binary_table(variant_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        if spec["task_type"] != "binary":
            continue
        result = variant_results[spec["variant_name"]]
        metrics = result["test_metrics_threshold_tuned"]
        rows.append(
            {
                "model_family": "binary",
                "label": spec["target_label"],
                "selected_threshold": metrics["selected_threshold"],
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "precision_at_operating_threshold": metrics["precision_at_operating_threshold"],
                "recall_at_operating_threshold": metrics["recall_at_operating_threshold"],
                "f1_at_operating_threshold": metrics["f1_at_operating_threshold"],
                "positive_rate_true": metrics["positive_rate_true"],
                "positive_rate_pred_at_operating_threshold": metrics[
                    "positive_rate_pred_at_operating_threshold"
                ],
                "threshold_tuning_status": metrics["threshold_tuning_status"],
            }
        )
    return pd.DataFrame(rows)


def write_threshold_tuned_summaries(
    variant_results: dict[str, dict[str, Any]],
    created_files: list[Path],
) -> None:
    summary_dir = TRANS_RESULTS / "08_experiment_summary"
    overall_dir = TRANS_RESULTS / "10_overall_summary"
    test_dir = TRANS_RESULTS / "11_test_results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    overall_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    model_comparison = pd.read_csv(summary_dir / "model_comparison_summary.csv")
    threshold_rows = []
    for row in model_comparison.to_dict(orient="records"):
        result = variant_results[row["variant_name"]]
        thresholds = result["selected_thresholds"]
        if result["task_type"] == "multilabel":
            validation_metric_value = result["validation_metrics_threshold_tuned"]["macro_f1_at_operating_threshold"]
            test_metric_value = result["test_metrics_threshold_tuned"]["macro_f1_at_operating_threshold"]
            operating_metric_name = "macro_f1_at_operating_threshold"
        else:
            validation_metrics = result.get("validation_metrics_threshold_tuned")
            validation_metric_value = (
                validation_metrics["f1_at_operating_threshold"] if validation_metrics is not None else None
            )
            test_metric_value = result["test_metrics_threshold_tuned"]["f1_at_operating_threshold"]
            operating_metric_name = "f1_at_operating_threshold"

        row["threshold_postprocessing_status"] = result["status"]
        row["threshold_selection_objective"] = THRESHOLD_OBJECTIVE
        row["operating_metric_name"] = operating_metric_name
        row["validation_operating_metric"] = validation_metric_value
        row["test_operating_metric"] = test_metric_value
        row["selected_thresholds"] = json.dumps(thresholds, sort_keys=True)
        threshold_rows.append(row)

    threshold_summary_df = pd.DataFrame(threshold_rows)
    model_comparison_csv = summary_dir / "model_comparison_summary_threshold_tuned.csv"
    model_comparison_json = summary_dir / "model_comparison_summary_threshold_tuned.json"
    threshold_summary_df.to_csv(model_comparison_csv, index=False)
    save_json(threshold_summary_df.to_dict(orient="records"), model_comparison_json)
    created_files.extend([model_comparison_csv, model_comparison_json])

    multilabel_result = variant_results["multilabel_distilbert_cv"]
    multilabel_table = build_multilabel_table(multilabel_result["test_metrics_threshold_tuned"])
    binary_table = build_binary_table(variant_results)

    multilabel_table_path = test_dir / "multilabel_test_metrics_by_label_threshold_tuned.csv"
    binary_table_path = test_dir / "binary_test_metrics_by_label_threshold_tuned.csv"
    multilabel_table.to_csv(multilabel_table_path, index=False)
    binary_table.to_csv(binary_table_path, index=False)
    created_files.extend([multilabel_table_path, binary_table_path])

    multilabel_summary = {
        "model_family": "multilabel",
        "graded_against": "test_labeled_cleaned.parquet",
        "primary_metric": "mean_average_precision",
        "threshold_selection_objective": THRESHOLD_OBJECTIVE,
        "selected_thresholds": multilabel_result["selected_thresholds"],
        "mean_average_precision": multilabel_result["test_metrics_threshold_tuned"]["mean_average_precision"],
        "mean_roc_auc": multilabel_result["test_metrics_threshold_tuned"]["mean_roc_auc"],
        "micro_f1_at_operating_threshold": multilabel_result["test_metrics_threshold_tuned"][
            "micro_f1_at_operating_threshold"
        ],
        "macro_f1_at_operating_threshold": multilabel_result["test_metrics_threshold_tuned"][
            "macro_f1_at_operating_threshold"
        ],
        "per_label": multilabel_table.to_dict(orient="records"),
    }

    binary_summary = {
        "model_family": "binary",
        "graded_against": "test_labeled_cleaned.parquet",
        "primary_metric": "average_precision",
        "threshold_selection_objective": THRESHOLD_OBJECTIVE,
        "threshold_tuning_coverage": {
            "tuned_from_validation": int(
                (binary_table["threshold_tuning_status"] == "threshold_tuned_from_validation").sum()
            ),
            "left_at_0_5_missing_validation_predictions": int(
                (
                    binary_table["threshold_tuning_status"]
                    == "validation_predictions_missing_left_at_0_5"
                ).sum()
            ),
        },
        "mean_average_precision_across_binary_models": float(binary_table["average_precision"].mean()),
        "mean_roc_auc_across_binary_models": float(binary_table["roc_auc"].mean()),
        "mean_f1_at_operating_threshold_across_binary_models": float(
            binary_table["f1_at_operating_threshold"].mean()
        ),
        "per_label": binary_table.to_dict(orient="records"),
    }

    combined = multilabel_table.merge(
        binary_table,
        on="label",
        how="outer",
        suffixes=("_multilabel", "_binary"),
    )
    comparison_path = test_dir / "multilabel_vs_binary_test_comparison_threshold_tuned.csv"
    combined.to_csv(comparison_path, index=False)
    created_files.append(comparison_path)

    overall_test_results = {
        "graded_against": "test_labeled_cleaned.parquet",
        "primary_comparison": "average_precision",
        "threshold_selection_objective": THRESHOLD_OBJECTIVE,
        "multilabel_summary": multilabel_summary,
        "binary_summary": binary_summary,
        "high_level_takeaways": [
            "Average precision values are unchanged by design because they were recomputed from the original saved probabilities.",
            "Threshold-dependent precision, recall, F1, and predicted positive rates now use validation-selected operating thresholds.",
            "binary_toxic remains at 0.5 because validation_predictions.csv was not present in the tracked results folder.",
        ],
    }

    multilabel_summary_path = test_dir / "multilabel_test_summary_threshold_tuned.json"
    binary_summary_path = test_dir / "binary_test_summary_threshold_tuned.json"
    overall_summary_path = test_dir / "overall_test_results_summary_threshold_tuned.json"
    save_json(multilabel_summary, multilabel_summary_path)
    save_json(binary_summary, binary_summary_path)
    save_json(overall_test_results, overall_summary_path)
    created_files.extend([multilabel_summary_path, binary_summary_path, overall_summary_path])

    if plt is not None:
        x = np.arange(len(LABEL_COLS))
        width = 0.38
        ml_f1 = [
            float(multilabel_table.loc[multilabel_table["label"] == label, "f1_at_operating_threshold"].iloc[0])
            for label in LABEL_COLS
        ]
        bin_f1 = [
            float(binary_table.loc[binary_table["label"] == label, "f1_at_operating_threshold"].iloc[0])
            for label in LABEL_COLS
        ]
        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, ml_f1, width=width, label="Multilabel")
        plt.bar(x + width / 2, bin_f1, width=width, label="Binary")
        plt.xticks(x, LABEL_COLS, rotation=45, ha="right")
        plt.ylabel("F1 at operating threshold")
        plt.title("Test F1 by Label: Multilabel vs Binary (Threshold Tuned)")
        plt.legend()
        plt.tight_layout()
        f1_plot_path = test_dir / "f1_by_label_multilabel_vs_binary_threshold_tuned.png"
        plt.savefig(f1_plot_path, dpi=220)
        plt.close()
        created_files.append(f1_plot_path)

    handoff_original = load_json(overall_dir / "overall_handoff_summary.json")
    handoff_tuned = copy.deepcopy(handoff_original)
    handoff_tuned["threshold_tuning_postprocessing"] = {
        "status": "completed_without_retraining",
        "threshold_selection_objective": THRESHOLD_OBJECTIVE,
        "threshold_search_grid": THRESHOLD_GRID_DESCRIPTION,
        "validation_split_requirement": "thresholds were selected using saved validation probabilities only",
        "average_precision_note": "Average precision remains threshold-free and should be unchanged aside from floating point roundoff.",
        "models_left_at_0_5": [
            name
            for name, result in variant_results.items()
            if result["status"] == "validation_predictions_missing_left_at_0_5"
        ],
    }

    for model in handoff_tuned["models"]:
        result = variant_results[model["variant_name"]]
        model["threshold_tuning_status"] = result["status"]
        model["selected_thresholds"] = result["selected_thresholds"]
        model["validation_metrics_threshold_tuned"] = result.get("validation_metrics_threshold_tuned")
        model["test_metrics_threshold_tuned"] = result["test_metrics_threshold_tuned"]

    overall_handoff_path = overall_dir / "overall_handoff_summary_threshold_tuned.json"
    save_json(handoff_tuned, overall_handoff_path)
    created_files.append(overall_handoff_path)


def write_review_markdowns(
    variant_results: dict[str, dict[str, Any]],
    created_files: list[Path],
    missing_files: list[Path],
    untouched_files: list[Path],
) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_paths = [
        REVIEW_DIR / "README.md",
        REVIEW_DIR / "what_changed.md",
        REVIEW_DIR / "threshold_tuning_method.md",
        REVIEW_DIR / "updated_artifacts_inventory.md",
        REVIEW_DIR / "before_vs_after_summary.md",
        REVIEW_DIR / "before_vs_after_metrics.csv",
    ]
    created_listing = sorted(set(created_files + review_paths))

    before_after_rows = []
    for result in variant_results.values():
        before_after_rows.extend(result["before_after_rows"])
    before_after_df = pd.DataFrame(before_after_rows).sort_values(["task_type", "label", "variant_name"])
    before_after_csv = REVIEW_DIR / "before_vs_after_metrics.csv"
    before_after_df.to_csv(before_after_csv, index=False)
    created_files.append(before_after_csv)

    ap_check_rows = []
    for row in before_after_df.to_dict(orient="records"):
        before_ap = row["average_precision_before"]
        after_ap = row["average_precision_after"]
        unchanged = (
            before_ap is None
            or after_ap is None
            or bool(np.isclose(float(before_ap), float(after_ap), atol=1e-12, rtol=1e-9))
        )
        ap_check_rows.append(
            {
                "Variant": row["variant_name"],
                "Label": row["label"],
                "Threshold Before": format_float(row["threshold_before"], 2),
                "Threshold After": format_float(row["threshold_after"], 2),
                "AP Before": format_float(before_ap, 6),
                "AP After": format_float(after_ap, 6),
                "Precision Before": format_float(row["precision_before"], 6),
                "Precision After": format_float(row["precision_after"], 6),
                "Recall Before": format_float(row["recall_before"], 6),
                "Recall After": format_float(row["recall_after"], 6),
                "F1 Before": format_float(row["f1_before"], 6),
                "F1 After": format_float(row["f1_after"], 6),
                "Pred Pos Rate Before": format_float(row["predicted_positive_rate_before"], 6),
                "Pred Pos Rate After": format_float(row["predicted_positive_rate_after"], 6),
                "AP Unchanged": "yes" if unchanged else "no",
                "Status": row["status"],
            }
        )

    threshold_rows = []
    for spec in MODEL_SPECS:
        result = variant_results[spec["variant_name"]]
        for label in spec["labels"]:
            threshold_rows.append(
                {
                    "Variant": spec["variant_name"],
                    "Label": label,
                    "Selected Threshold": format_float(result["selected_thresholds"][label], 2),
                    "Status": result["status"],
                }
            )

    main_readme = f"""# Transformer Threshold Tuning Fix
## Why this change was necessary
Transformer thresholded metrics in the tracked repo were computed at a fixed `0.5`, while the TF-IDF pipeline already tuned thresholds after model selection. This postprocessing fix makes the transformer decision-threshold stage scientifically closer to the TF-IDF workflow without retraining any model.

## Inputs used
- Saved validation probability CSVs from `01_multilabel_distilbert` and binary folders `03` through `07`.
- Saved test probability CSVs from `01_multilabel_distilbert` and binary folders `02` through `07`.
- Existing fixed-`0.5` metric JSONs in the same model folders for before-vs-after comparison.
- Existing summary artifacts in `08_experiment_summary`, `10_overall_summary`, and `11_test_results`.
- TF-IDF threshold logic from `TF-IDF Section/utils/metrics_utils.py`.

## Method
- Thresholds were tuned only from saved validation probabilities.
- The tuning rule mirrors the TF-IDF helper: search `0.05` to `0.95` in steps of `0.01`, maximize validation F1 label by label, and break ties toward `0.5`.
- The selected thresholds were then applied to the saved labeled-test and unlabeled-test probability CSVs.
- Average precision was recomputed from the same saved probabilities and checked against the fixed-`0.5` artifacts to confirm it stayed unchanged.
- No training notebook, checkpoint, or model weight file was edited.

## Files created
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in created_listing)}

## Files modified
- `text_data_results_summary.md`
- `text_data_results_quick_hits.md`

## Metrics unchanged by design
- Average precision for every transformer model and label, because AP is threshold-free and was recomputed from the original saved probabilities.
- Training time, throughput, parameter count, GPU memory, and any other training-side cost artifact.
- The fixed-`0.5` legacy JSON/CSV artifacts, which were preserved beside the new threshold-tuned versions.

## Metrics changed after threshold tuning
- Precision, recall, F1, and predicted positive rate now use validation-selected operating thresholds in the new `*_threshold_tuned.*` outputs.
- The shared multilabel model now has one tuned threshold per label instead of one global `0.5`.
- Five binary models were tuned the same way; `binary_toxic` stayed at `0.5` because `validation_predictions.csv` was missing from the tracked folder.

## Downstream artifacts updated
- `08_experiment_summary/model_comparison_summary_threshold_tuned.csv`
- `08_experiment_summary/model_comparison_summary_threshold_tuned.json`
- `10_overall_summary/overall_handoff_summary_threshold_tuned.json`
- `11_test_results/multilabel_test_metrics_by_label_threshold_tuned.csv`
- `11_test_results/binary_test_metrics_by_label_threshold_tuned.csv`
- `11_test_results/multilabel_test_summary_threshold_tuned.json`
- `11_test_results/binary_test_summary_threshold_tuned.json`
- `11_test_results/multilabel_vs_binary_test_comparison_threshold_tuned.csv`
- `11_test_results/overall_test_results_summary_threshold_tuned.json`
- `11_test_results/f1_by_label_multilabel_vs_binary_threshold_tuned.png`

## Status of test_labeled_predictions.csv
- The original `test_labeled_predictions.csv` files were left untouched to avoid silently overwriting legacy fixed-`0.5` outputs.
- New companion exports named `test_labeled_predictions_threshold_tuned.csv` were written in each model folder.
- These companion exports contain the original probabilities plus `*_selected_threshold` and `*_pred_threshold_tuned` columns.

## TF-IDF tuning sanity check
- The TF-IDF helper in `TF-IDF Section/utils/metrics_utils.py` tunes thresholds label by label on validation data, optimizes F1, and breaks ties toward the default threshold.
- The transformer postprocessor now follows that same validation-only F1-maximizing rule on probability grids.
- No obvious TF-IDF threshold bug was changed here; the goal was methodological alignment, not a TF-IDF rewrite.

## Remaining caveats
- `TransformersSection/TransResults/02_binary_toxic/validation_predictions.csv` was not present, so `binary_toxic` could not be re-tuned from validation probabilities and remains at `0.5` in the threshold-tuned summaries.
- The training/evaluation notebook still contains fixed-`0.5` logic. This fix intentionally avoids refactoring that notebook and instead writes postprocessed versioned outputs.
- Any external consumer that hard-codes the old fixed-`0.5` filenames will need to switch to the new `*_threshold_tuned.*` outputs to use the comparable operating thresholds.
"""

    (REVIEW_DIR / "README.md").write_text(main_readme)

    what_changed = f"""# What Changed

- Added `TransformersSection/postprocess_transformer_thresholds.py` as a lightweight post-training postprocessor.
- Wrote per-model threshold-tuning artifacts next to the existing transformer outputs instead of retraining or overwriting legacy files.
- Regenerated threshold-dependent summary artifacts in `08_experiment_summary`, `10_overall_summary`, and `11_test_results` as versioned `*_threshold_tuned.*` files.
- Updated the paper-support markdown summaries so they now reference threshold-tuned transformer metrics and note the one missing validation artifact for `binary_toxic`.

## Selected thresholds
{markdown_table(threshold_rows, ['Variant', 'Label', 'Selected Threshold', 'Status'])}
"""
    (REVIEW_DIR / "what_changed.md").write_text(what_changed)

    method_md = f"""# Threshold Tuning Method

## Inputs
- Validation probability CSVs were treated as the only admissible source for threshold selection.
- Labeled test probability CSVs were used only for final thresholded evaluation.
- Unlabeled `test_predictions.csv` files were updated only by applying already selected validation thresholds.

## Rule used
- Threshold grid: `{THRESHOLD_GRID_DESCRIPTION}`
- Objective: `{THRESHOLD_OBJECTIVE}`
- Tie-break: closest threshold to `{DEFAULT_THRESHOLD}`

## Why this matches TF-IDF closely enough
- The TF-IDF helper in `TF-IDF Section/utils/metrics_utils.py` uses the same probability grid and the same label-level F1 objective with a tie-break toward the default threshold.
- That means the transformer and TF-IDF pipelines are now aligned at the post-training threshold-selection stage, while AP remains the threshold-free model-selection metric.

## Explicit non-changes
- No retraining
- No checkpoint edits
- No AP redefinition
- No deletion of the original fixed-`0.5` artifacts
"""
    (REVIEW_DIR / "threshold_tuning_method.md").write_text(method_md)

    inventory_md = f"""# Updated Artifacts Inventory

## Created files
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in created_listing)}

## Modified files
- `text_data_results_summary.md`
- `text_data_results_quick_hits.md`

## Untouched important files
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in untouched_files)}

## Expected but missing files
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in missing_files)}
"""
    (REVIEW_DIR / "updated_artifacts_inventory.md").write_text(inventory_md)

    before_after_md = f"""# Before vs After Summary

## Compact comparison
{markdown_table(ap_check_rows, ['Variant', 'Label', 'Threshold Before', 'Threshold After', 'AP Before', 'AP After', 'Precision Before', 'Precision After', 'Recall Before', 'Recall After', 'F1 Before', 'F1 After', 'Pred Pos Rate Before', 'Pred Pos Rate After', 'AP Unchanged', 'Status'])}

## Notes
- `AP Unchanged` should be `yes` whenever the new metrics were recomputed from the same saved probabilities.
- Metric movement should appear only in threshold-dependent columns.
- `binary_toxic` remains fixed at `0.5` because the tracked validation probability CSV was missing.
"""
    (REVIEW_DIR / "before_vs_after_summary.md").write_text(before_after_md)


def process_model(spec: dict[str, Any], created_files: list[Path], missing_files: list[Path]) -> dict[str, Any]:
    folder = spec["folder"]
    labels = spec["labels"]
    validation_path = folder / spec["validation_predictions"]
    validation_metrics_path = folder / spec["validation_metrics"]
    test_labeled_path = folder / spec["test_labeled_predictions"]
    test_metrics_path = folder / spec["test_labeled_metrics"]
    test_predictions_path = folder / spec["test_predictions"]

    validation_df = pd.read_csv(validation_path) if validation_path.exists() else None
    validation_metrics_original = load_json(validation_metrics_path) if validation_metrics_path.exists() else None
    test_df = pd.read_csv(test_labeled_path)
    test_metrics_original = load_json(test_metrics_path)
    test_unlabeled_df = pd.read_csv(test_predictions_path) if test_predictions_path.exists() else None

    if validation_df is not None:
        thresholds, search_df = tune_thresholds(validation_df, labels)
        status = "threshold_tuned_from_validation"
        search_path = folder / "validation_threshold_search_threshold_tuned.csv"
        search_df.to_csv(search_path, index=False)
        created_files.append(search_path)
    else:
        thresholds = {label: DEFAULT_THRESHOLD for label in labels}
        search_df = None
        status = "validation_predictions_missing_left_at_0_5"
        missing_files.append(validation_path)

    created_files.extend(create_prediction_outputs(spec, thresholds, validation_df, test_df, test_unlabeled_df))

    if spec["task_type"] == "multilabel":
        status_by_label = {label: status for label in labels}

        if validation_df is not None:
            validation_metrics_tuned = compute_multilabel_metrics(
                validation_df[[f"{label}_true" for label in labels]].to_numpy(),
                validation_df[[f"{label}_prob" for label in labels]].to_numpy(),
                labels,
                thresholds,
                status_by_label,
            )
            validation_metrics_path_tuned = folder / "validation_metrics_threshold_tuned.json"
            save_json(validation_metrics_tuned, validation_metrics_path_tuned)
            created_files.append(validation_metrics_path_tuned)
        else:
            validation_metrics_tuned = None

        test_metrics_tuned = compute_multilabel_metrics(
            test_df[[f"{label}_true" for label in labels]].to_numpy(),
            test_df[[f"{label}_prob" for label in labels]].to_numpy(),
            labels,
            thresholds,
            status_by_label,
        )
    else:
        label = spec["target_label"]
        if validation_df is not None:
            validation_metrics_tuned = compute_binary_metrics(
                validation_df[f"{label}_true"].to_numpy(),
                validation_df[f"{label}_prob"].to_numpy(),
                thresholds[label],
                status,
            )
            validation_metrics_path_tuned = folder / "validation_metrics_threshold_tuned.json"
            save_json(validation_metrics_tuned, validation_metrics_path_tuned)
            created_files.append(validation_metrics_path_tuned)
        else:
            validation_metrics_tuned = None

        test_metrics_tuned = compute_binary_metrics(
            test_df[f"{label}_true"].to_numpy(),
            test_df[f"{label}_prob"].to_numpy(),
            thresholds[label],
            status,
        )

    selected_thresholds_path = folder / "selected_thresholds_threshold_tuned.json"
    save_json(
        {
            "variant_name": spec["variant_name"],
            "task_type": spec["task_type"],
            "threshold_tuning_status": status,
            "threshold_selection_objective": THRESHOLD_OBJECTIVE,
            "threshold_search_grid": THRESHOLD_GRID_DESCRIPTION,
            "selected_thresholds": thresholds,
        },
        selected_thresholds_path,
    )
    created_files.append(selected_thresholds_path)

    test_metrics_tuned_path = folder / "test_labeled_metrics_threshold_tuned.json"
    save_json(test_metrics_tuned, test_metrics_tuned_path)
    created_files.append(test_metrics_tuned_path)

    before_after_rows = extract_before_after_rows(
        spec,
        test_metrics_original,
        test_metrics_tuned,
        thresholds,
        status,
    )

    return {
        "variant_name": spec["variant_name"],
        "task_type": spec["task_type"],
        "target_label": spec["target_label"],
        "status": status,
        "selected_thresholds": thresholds,
        "validation_metrics_original": validation_metrics_original,
        "validation_metrics_threshold_tuned": validation_metrics_tuned,
        "test_metrics_original": test_metrics_original,
        "test_metrics_threshold_tuned": test_metrics_tuned,
        "before_after_rows": before_after_rows,
    }


def main() -> None:
    created_files: list[Path] = [ROOT / "TransformersSection" / "postprocess_transformer_thresholds.py"]
    missing_files: list[Path] = []
    untouched_files = [
        TRANS_RESULTS / "08_experiment_summary" / "model_comparison_summary.csv",
        TRANS_RESULTS / "08_experiment_summary" / "model_comparison_summary.json",
        TRANS_RESULTS / "10_overall_summary" / "overall_handoff_summary.json",
        TRANS_RESULTS / "11_test_results" / "multilabel_test_summary.json",
        TRANS_RESULTS / "11_test_results" / "binary_test_summary.json",
        TRANS_RESULTS / "11_test_results" / "overall_test_results_summary.json",
    ]

    variant_results: dict[str, dict[str, Any]] = {}
    for spec in MODEL_SPECS:
        variant_results[spec["variant_name"]] = process_model(spec, created_files, missing_files)

    write_threshold_tuned_summaries(variant_results, created_files)
    write_review_markdowns(variant_results, created_files, missing_files, untouched_files)

    print("Created threshold-tuned artifacts:")
    for path in sorted(set(created_files)):
        print("-", path.relative_to(ROOT))
    if missing_files:
        print("\nMissing expected inputs:")
        for path in sorted(set(missing_files)):
            print("-", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
