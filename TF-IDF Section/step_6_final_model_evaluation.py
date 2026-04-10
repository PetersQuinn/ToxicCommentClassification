from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import build_config, comparison_dir, ensure_project_dirs
from utils.io_utils import load_json, save_frame
from utils.metrics_utils import selection_values
from utils.modeling_utils import (
    evaluate_model,
    feature_attributions,
    fit_final_model,
    save_model,
    scored_output_frame,
)
from utils.pipeline_utils import (
    add_model_args,
    approach_output_dir,
    comparison_frame_from_metrics,
    load_labeled_data,
    resolve_model_names,
    save_comparison_bundle,
    save_metric_bundle,
)
from utils.plotting_utils import plot_curve_grid, plot_threshold_curve, plot_top_features
from utils.progress_utils import build_step_logger


def _load_tuning_artifacts(root):
    best_params_path = root / "best_params.json"
    best_thresholds_path = root / "best_thresholds.json"
    if not best_params_path.exists() or not best_thresholds_path.exists():
        raise FileNotFoundError(f"Missing tuning artifacts under {root}")
    best_params = load_json(best_params_path, default={})
    for key, value in list(best_params.items()):
        if "ngram_range" in key and isinstance(value, list):
            best_params[key] = tuple(value)
    best_thresholds = load_json(best_thresholds_path, default={})
    return best_params, best_thresholds


def _selection_tuple(frame: pd.DataFrame, config) -> tuple[float, ...]:
    metric_names = [config.selection_metric] + list(config.selection_tie_break_metrics)
    return selection_values(frame, metric_names)


def _is_better_selection(candidate: tuple[float, ...], incumbent: tuple[float, ...] | None) -> bool:
    if incumbent is None:
        return True
    for candidate_value, incumbent_value in zip(candidate, incumbent):
        candidate_missing = np.isnan(candidate_value)
        incumbent_missing = np.isnan(incumbent_value)
        if candidate_missing and incumbent_missing:
            continue
        if incumbent_missing and not candidate_missing:
            return True
        if candidate_missing and not incumbent_missing:
            return False
        if candidate_value > incumbent_value:
            return True
        if candidate_value < incumbent_value:
            return False
    return False


def _metric_value(frame: pd.DataFrame, metric_name: str) -> float:
    if frame.empty:
        return float("nan")
    match = frame.loc[frame["metric"] == metric_name, "value"]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def _label_metric_value(frame: pd.DataFrame, label_name: str, metric_name: str) -> float:
    if frame.empty:
        return float("nan")
    match = frame.loc[frame["label"] == label_name, metric_name]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def _identity_hate_export_frame(source_frame: pd.DataFrame, pred_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": source_frame["id"].to_numpy(),
            "identity_hate_pred": pred_frame["identity_hate"].astype(int).to_numpy(),
        }
    )


def _save_identity_hate_export(
    root: Path,
    source_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    logger,
    indent: int = 0,
) -> tuple[pd.DataFrame, Path]:
    export_frame = _identity_hate_export_frame(source_frame, pred_frame)
    export_path = root / "test_identity_hate_predictions.parquet"
    save_frame(export_frame, export_path, index=False)
    logger.saved(export_path, indent=indent)
    return export_frame, export_path


def _save_identity_hate_export_summary(
    config,
    summary_frame: pd.DataFrame,
    logger,
) -> None:
    csv_path = config.outputs_dir / "final_identity_hate_prediction_exports_summary.csv"
    save_frame(summary_frame, csv_path, index=False)
    logger.saved(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit tuned models and evaluate them on the labeled test split.")
    add_model_args(parser)
    parser.add_argument(
        "--identity-hate-export-only",
        action="store_true",
        help="Fit tuned models only far enough to export test-set identity_hate predictions and the best-AP summary.",
    )
    args = parser.parse_args()

    config = build_config()
    ensure_project_dirs(config)
    logger = build_step_logger(config, "step_6")
    logger.info("Starting final model evaluation")
    try:
        train, test_labeled, text_column = load_labeled_data(config)
        model_names = resolve_model_names(args.models)
        logger.event(
            "Loaded train and labeled test splits",
            train_rows=len(train),
            test_rows=len(test_labeled),
            text_column=text_column,
            model_selection_source="retained_model_families",
            models=", ".join(model_names),
            export_only=args.identity_hate_export_only,
        )

        identity_export_rows: list[dict[str, object]] = []
        best_model_name: str | None = None
        best_selection_tuple: tuple[float, ...] | None = None
        best_metric_value = float("nan")
        best_export_frame = pd.DataFrame()
        best_export_path: Path | None = None

        for model_index, model_name in enumerate(model_names, start=1):
            logger.event(f"Model {model_index}/{len(model_names)} evaluation started", model_name=model_name)

            ml_root = approach_output_dir(config, model_name, "multilabel")
            ml_params, ml_thresholds = _load_tuning_artifacts(ml_root)
            ml_model = fit_final_model(
                train,
                config.label_cols,
                text_column,
                model_name,
                "multilabel",
                config,
                ml_params,
                logger=logger,
                run_label=f"{model_name} multilabel final fit",
                indent=1,
            )
            ml_eval = evaluate_model(
                ml_model,
                test_labeled,
                config.label_cols,
                text_column,
                ml_thresholds,
                logger=logger,
                run_label=f"{model_name} multilabel final evaluation",
                indent=1,
            )
            ml_export_frame, ml_export_path = _save_identity_hate_export(
                ml_root,
                test_labeled,
                ml_eval["pred"],
                logger,
                indent=1,
            )
            ml_metric_value = _metric_value(ml_eval["aggregate_metrics"], config.selection_metric)
            ml_identity_pr_auc = _label_metric_value(ml_eval["label_metrics"], "identity_hate", "pr_auc")
            identity_export_rows.append(
                {
                    "model_name": model_name,
                    "evaluation_path": "multilabel",
                    "metric_name": config.selection_metric,
                    "metric_value": ml_metric_value,
                    "identity_hate_pr_auc": ml_identity_pr_auc,
                    "rows": len(ml_export_frame),
                    "export_path": logger.relative_path(ml_export_path),
                    "test_split": "test_labeled",
                }
            )
            candidate_selection_tuple = _selection_tuple(ml_eval["aggregate_metrics"], config)
            if _is_better_selection(candidate_selection_tuple, best_selection_tuple):
                best_model_name = model_name
                best_selection_tuple = candidate_selection_tuple
                best_metric_value = ml_metric_value
                best_export_frame = ml_export_frame.copy()
                best_export_path = ml_export_path

            if not args.identity_hate_export_only:
                ml_threshold_frame = pd.read_csv(ml_root / "threshold_summary.csv") if (ml_root / "threshold_summary.csv").exists() else None
                save_metric_bundle(
                    root=ml_root,
                    label_metrics=ml_eval["label_metrics"],
                    aggregate_metrics=ml_eval["aggregate_metrics"],
                    threshold_frame=ml_threshold_frame,
                    logger=logger,
                )
                save_frame(ml_eval["error_samples"], ml_root / "error_analysis_samples.csv", index=False)
                save_frame(
                    scored_output_frame(test_labeled, ml_eval["scores"], ml_eval["pred"], ml_thresholds, config.label_cols, text_column),
                    ml_root / "scored_test_predictions.csv",
                    index=False,
                )
                save_model(ml_model, ml_root / "saved_model" / "model.joblib")
                logger.saved(ml_root / "error_analysis_samples.csv", indent=1)
                logger.saved(ml_root / "scored_test_predictions.csv", indent=1)
                logger.saved(ml_root / "saved_model" / "model.joblib", indent=1)
                plot_curve_grid(
                    test_labeled[config.label_cols].to_numpy(),
                    ml_eval["scores"],
                    config.label_cols,
                    ml_root / "plots" / "pr_curves.png",
                    curve_type="pr",
                )
                plot_curve_grid(
                    test_labeled[config.label_cols].to_numpy(),
                    ml_eval["scores"],
                    config.label_cols,
                    ml_root / "plots" / "roc_curves.png",
                    curve_type="roc",
                )
                logger.saved(ml_root / "plots" / "pr_curves.png", indent=1)
                logger.saved(ml_root / "plots" / "roc_curves.png", indent=1)
                if ml_threshold_frame is not None:
                    for label_name in config.label_cols:
                        plot_threshold_curve(ml_threshold_frame, label_name, ml_root / "plots" / f"threshold_curve_{label_name}.png")
                        logger.saved(ml_root / "plots" / f"threshold_curve_{label_name}.png", indent=2)
                ml_features = feature_attributions(ml_model, config.label_cols)
                if not ml_features.empty:
                    save_frame(ml_features, ml_root / "feature_coefficients.csv", index=False)
                    logger.saved(ml_root / "feature_coefficients.csv", indent=1)
                    for label_name in config.label_cols:
                        plot_top_features(ml_features, label_name, "positive", ml_root / "plots" / f"top_positive_features_{label_name}.png")
                        plot_top_features(ml_features, label_name, "negative", ml_root / "plots" / f"top_negative_features_{label_name}.png")
                        logger.saved(ml_root / "plots" / f"top_positive_features_{label_name}.png", indent=2)
                        logger.saved(ml_root / "plots" / f"top_negative_features_{label_name}.png", indent=2)

            binary_metric_frames = []
            binary_labels = ["identity_hate"] if args.identity_hate_export_only else list(config.label_cols)
            for label_index, label_name in enumerate(binary_labels, start=1):
                logger.event(
                    f"Binary evaluation {label_index}/{len(binary_labels)} started",
                    indent=1,
                    label=label_name,
                )
                binary_root = approach_output_dir(config, model_name, "binary", label_name)
                binary_params, binary_thresholds = _load_tuning_artifacts(binary_root)
                binary_model = fit_final_model(
                    train,
                    [label_name],
                    text_column,
                    model_name,
                    "binary",
                    config,
                    binary_params,
                    logger=logger,
                    run_label=f"{model_name} binary final fit {label_name}",
                    indent=2,
                )
                binary_eval = evaluate_model(
                    binary_model,
                    test_labeled,
                    [label_name],
                    text_column,
                    binary_thresholds,
                    logger=logger,
                    run_label=f"{model_name} binary final evaluation {label_name}",
                    indent=2,
                )

                if label_name == "identity_hate":
                    binary_export_frame, binary_export_path = _save_identity_hate_export(
                        binary_root,
                        test_labeled,
                        binary_eval["pred"],
                        logger,
                        indent=2,
                    )
                    identity_export_rows.append(
                        {
                            "model_name": model_name,
                            "evaluation_path": "binary_identity_hate",
                            "metric_name": "pr_auc",
                            "metric_value": _label_metric_value(binary_eval["label_metrics"], "identity_hate", "pr_auc"),
                            "identity_hate_pr_auc": _label_metric_value(binary_eval["label_metrics"], "identity_hate", "pr_auc"),
                            "rows": len(binary_export_frame),
                            "export_path": logger.relative_path(binary_export_path),
                            "test_split": "test_labeled",
                        }
                    )

                if not args.identity_hate_export_only:
                    binary_threshold_frame = pd.read_csv(binary_root / "threshold_summary.csv") if (binary_root / "threshold_summary.csv").exists() else None
                    save_metric_bundle(
                        root=binary_root,
                        label_metrics=binary_eval["label_metrics"],
                        aggregate_metrics=binary_eval["aggregate_metrics"],
                        threshold_frame=binary_threshold_frame,
                        logger=logger,
                    )
                    save_frame(binary_eval["error_samples"], binary_root / "error_analysis_samples.csv", index=False)
                    save_frame(
                        scored_output_frame(test_labeled, binary_eval["scores"], binary_eval["pred"], binary_thresholds, [label_name], text_column),
                        binary_root / "scored_test_predictions.csv",
                        index=False,
                    )
                    save_model(binary_model, binary_root / "saved_model" / "model.joblib")
                    logger.saved(binary_root / "error_analysis_samples.csv", indent=2)
                    logger.saved(binary_root / "scored_test_predictions.csv", indent=2)
                    logger.saved(binary_root / "saved_model" / "model.joblib", indent=2)
                    plot_curve_grid(
                        test_labeled[[label_name]].to_numpy(),
                        binary_eval["scores"],
                        [label_name],
                        binary_root / "plots" / "pr_curve.png",
                        curve_type="pr",
                    )
                    plot_curve_grid(
                        test_labeled[[label_name]].to_numpy(),
                        binary_eval["scores"],
                        [label_name],
                        binary_root / "plots" / "roc_curve.png",
                        curve_type="roc",
                    )
                    logger.saved(binary_root / "plots" / "pr_curve.png", indent=2)
                    logger.saved(binary_root / "plots" / "roc_curve.png", indent=2)
                    if binary_threshold_frame is not None:
                        plot_threshold_curve(binary_threshold_frame, label_name, binary_root / "plots" / "threshold_curve.png")
                        logger.saved(binary_root / "plots" / "threshold_curve.png", indent=2)
                    binary_features = feature_attributions(binary_model, [label_name])
                    if not binary_features.empty:
                        save_frame(binary_features, binary_root / "feature_coefficients.csv", index=False)
                        plot_top_features(binary_features, label_name, "positive", binary_root / "plots" / "top_positive_features.png")
                        plot_top_features(binary_features, label_name, "negative", binary_root / "plots" / "top_negative_features.png")
                        logger.saved(binary_root / "feature_coefficients.csv", indent=2)
                        logger.saved(binary_root / "plots" / "top_positive_features.png", indent=2)
                        logger.saved(binary_root / "plots" / "top_negative_features.png", indent=2)
                    binary_metric_frames.append(binary_eval["label_metrics"])

                del binary_model, binary_eval
                gc.collect()

            if not args.identity_hate_export_only and binary_metric_frames:
                binary_metrics = pd.concat(binary_metric_frames, ignore_index=True)
                comparison_frame = comparison_frame_from_metrics(ml_eval["label_metrics"], binary_metrics)
                save_comparison_bundle(
                    comparison_dir(config, model_name),
                    comparison_frame,
                    logger=logger,
                )

            logger.event(
                "Model evaluation complete",
                model_name=model_name,
                macro_pr_auc=ml_metric_value,
                identity_hate_pr_auc=ml_identity_pr_auc,
            )
            del ml_model, ml_eval
            gc.collect()

        if best_model_name is None or best_export_path is None or best_export_frame.empty:
            raise RuntimeError("No best-AP model could be identified from the final evaluation outputs.")

        summary_frame = pd.DataFrame(identity_export_rows)
        summary_frame["is_best_ap_model_family"] = summary_frame["model_name"] == best_model_name
        summary_frame["is_primary_best_ap_export"] = (
            (summary_frame["model_name"] == best_model_name)
            & (summary_frame["evaluation_path"] == "multilabel")
        )
        best_copy_path = config.outputs_dir / "best_ap_identity_hate_test_predictions.parquet"
        save_frame(best_export_frame, best_copy_path, index=False)
        logger.saved(best_copy_path)
        _save_identity_hate_export_summary(
            config=config,
            summary_frame=summary_frame,
            logger=logger,
        )
        logger.event(
            "Best AP model resolved",
            model_name=best_model_name,
            selection_metric=config.selection_metric,
            selection_value=best_metric_value,
            export_path=logger.relative_path(best_copy_path),
        )
        logger.event(
            "Final evaluation complete",
            elapsed=logger.elapsed(),
            best_ap_model=best_model_name,
            best_ap_value=best_metric_value,
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
