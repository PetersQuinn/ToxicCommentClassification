from __future__ import annotations

import argparse
import gc
from pathlib import Path

from utils.config import STEP4_REFERENCE_TFIDF, build_config, ensure_project_dirs
from utils.io_utils import format_storage_size, load_json, remove_file, save_frame, save_json
from utils.metrics_utils import metric_lookup
from utils.modeling_utils import (
    build_binary_pipeline,
    build_multilabel_pipeline,
    run_cv_experiment,
    scored_output_frame,
    sample_training_fraction,
    step4_tfidf_overrides,
)
from utils.pipeline_utils import add_model_args, approach_output_dir, load_labeled_data, resolve_model_names
from utils.progress_utils import build_step_logger

LARGE_PRETUNING_FILENAMES = [
    "baseline_cv_scored_samples.csv",
    "baseline_cv_scored_samples.parquet",
]
COMPACT_OUTPUT_FILENAMES = [
    "baseline_cv_fold_metrics.csv",
    "baseline_cv_fold_label_metrics.csv",
    "baseline_cv_oof_aggregate_metrics.csv",
    "baseline_cv_oof_label_metrics.csv",
    "baseline_cv_threshold_diagnostics.csv",
    "baseline_cv_error_analysis_samples.csv",
]
RUN_CONFIG_FILENAME = "baseline_cv_run_config.json"


def _pretuning_roots(config, model_name: str) -> list[Path]:
    roots = [approach_output_dir(config, model_name, "multilabel") / "pretuning_results"]
    for label_name in config.label_cols:
        roots.append(approach_output_dir(config, model_name, "binary", label_name) / "pretuning_results")
    return roots


def _cleanup_large_pretuning_artifacts(config, model_names: list[str], logger) -> None:
    logger.event(
        "Cleanup started",
        models=", ".join(model_names),
        targets=", ".join(LARGE_PRETUNING_FILENAMES),
    )
    total_deleted = 0
    total_bytes = 0
    scanned_roots = 0

    for model_name in model_names:
        model_deleted = 0
        model_bytes = 0
        roots = _pretuning_roots(config, model_name)
        logger.event(
            "Scanning pretuning roots",
            indent=1,
            model_name=model_name,
            roots=len(roots),
        )
        for root in roots:
            scanned_roots += 1
            for file_name in LARGE_PRETUNING_FILENAMES:
                candidate = root / file_name
                if not candidate.exists():
                    continue
                removed_bytes = remove_file(candidate)
                model_deleted += 1
                model_bytes += removed_bytes
                total_deleted += 1
                total_bytes += removed_bytes
                logger.event(
                    "Deleted large pretuning artifact",
                    indent=2,
                    file=logger.relative_path(candidate),
                    size=format_storage_size(removed_bytes),
                )
        if model_deleted == 0:
            logger.info("No large row-level pretuning artifacts found", indent=2)
        else:
            logger.event(
                "Model cleanup summary",
                indent=2,
                model_name=model_name,
                deleted_files=model_deleted,
                freed=format_storage_size(model_bytes),
            )

    logger.event(
        "Cleanup finished",
        scanned_roots=scanned_roots,
        deleted_files=total_deleted,
        freed=format_storage_size(total_bytes),
        preserved="summary metrics, thresholds, markdown, plots",
    )


def _remove_stale_full_scored_artifacts(root: Path, logger, indent: int) -> None:
    for file_name in LARGE_PRETUNING_FILENAMES:
        candidate = root / file_name
        if not candidate.exists():
            continue
        removed_bytes = remove_file(candidate)
        logger.event(
            "Removed stale full scored artifact",
            indent=indent,
            file=logger.relative_path(candidate),
            size=format_storage_size(removed_bytes),
        )


def _expected_unit_paths(root: Path, save_full_scored: bool) -> list[Path]:
    paths = [root / file_name for file_name in COMPACT_OUTPUT_FILENAMES]
    paths.append(root / RUN_CONFIG_FILENAME)
    if save_full_scored:
        paths.append(root / "baseline_cv_scored_samples.parquet")
    return paths


def _build_run_signature(
    *,
    model_name: str,
    mode: str,
    label_name: str | None,
    train_fraction: float,
    sampled_rows: int,
    source_rows: int,
    text_column: str,
    save_full_scored: bool,
    config,
    tfidf_params: dict,
) -> dict:
    serializable_tfidf = {}
    for key, value in tfidf_params.items():
        if isinstance(value, tuple):
            serializable_tfidf[key] = list(value)
        else:
            serializable_tfidf[key] = str(value) if key == "dtype" else value
    return {
        "profile_name": "step4_tonight_compact_cv",
        "model_name": model_name,
        "mode": mode,
        "label_name": label_name,
        "train_fraction": float(train_fraction),
        "sampled_rows": int(sampled_rows),
        "source_rows": int(source_rows),
        "cv_folds": int(config.cv_folds),
        "text_column": text_column,
        "selection_metric": config.selection_metric,
        "selection_metric_label": config.selection_metric_label,
        "tfidf_profile_name": STEP4_REFERENCE_TFIDF["name"],
        "tfidf_params": serializable_tfidf,
        "save_full_scored": bool(save_full_scored),
        "compact_mode": True,
    }


def _unit_is_complete(root: Path, run_signature: dict, save_full_scored: bool) -> tuple[bool, str]:
    missing = [path.name for path in _expected_unit_paths(root, save_full_scored) if not path.exists()]
    if missing:
        return False, f"missing_outputs={','.join(missing)}"
    saved_signature = load_json(root / RUN_CONFIG_FILENAME, default=None)
    if saved_signature is None:
        return False, "missing_run_config"
    if saved_signature != run_signature:
        return False, "run_config_mismatch"
    return True, "ready"


def _save_cv_outputs(
    *,
    cv_result: dict,
    root: Path,
    source_frame,
    target_cols: list[str],
    text_column: str,
    save_full_scored: bool,
    logger,
    indent: int,
) -> None:
    save_frame(cv_result["fold_metrics"], root / "baseline_cv_fold_metrics.csv", index=False)
    save_frame(cv_result["fold_label_metrics"], root / "baseline_cv_fold_label_metrics.csv", index=False)
    save_frame(cv_result["oof_aggregate_metrics"], root / "baseline_cv_oof_aggregate_metrics.csv", index=False)
    save_frame(cv_result["oof_label_metrics"], root / "baseline_cv_oof_label_metrics.csv", index=False)
    save_frame(cv_result["threshold_frame"], root / "baseline_cv_threshold_diagnostics.csv", index=False)
    save_frame(cv_result["error_samples"], root / "baseline_cv_error_analysis_samples.csv", index=False)
    logger.saved(root / "baseline_cv_fold_metrics.csv", indent=indent)
    logger.saved(root / "baseline_cv_fold_label_metrics.csv", indent=indent)
    logger.saved(root / "baseline_cv_oof_aggregate_metrics.csv", indent=indent)
    logger.saved(root / "baseline_cv_oof_label_metrics.csv", indent=indent)
    logger.saved(root / "baseline_cv_threshold_diagnostics.csv", indent=indent)
    logger.saved(root / "baseline_cv_error_analysis_samples.csv", indent=indent)

    if not save_full_scored:
        _remove_stale_full_scored_artifacts(root, logger, indent=indent)
        logger.info("Skipped full OOF scored export in compact step_4 mode", indent=indent)
        return

    scored_frame = scored_output_frame(
        source_frame=source_frame,
        scores=cv_result["oof_scores"],
        pred=cv_result["oof_pred"],
        thresholds=cv_result["thresholds"],
        target_cols=target_cols,
        text_column=text_column,
    )
    save_frame(scored_frame, root / "baseline_cv_scored_samples.parquet", index=False)
    logger.saved(root / "baseline_cv_scored_samples.parquet", indent=indent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run untuned cross-validation and save error analysis outputs.")
    add_model_args(parser)
    parser.add_argument(
        "--cleanup-large-pretuning",
        action="store_true",
        help="Delete oversized old row-level pretuning artifacts before rerunning step 4.",
    )
    parser.add_argument(
        "--save-full-scored",
        action="store_true",
        help="Save full OOF scored outputs as Parquet in addition to the compact summaries.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="Optional fraction of the training split to use for step 4 baseline CV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed step-4 units instead of resuming from matching outputs.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Explicitly keep compact step-4 saving enabled. This is already the default.",
    )
    args = parser.parse_args()

    config = build_config()
    ensure_project_dirs(config)
    train_fraction = config.step4_default_train_fraction if args.train_fraction is None else args.train_fraction
    if train_fraction <= 0.0 or train_fraction > 1.0:
        raise ValueError("--train-fraction must be in the interval (0, 1].")
    logger = build_step_logger(config, "step_4")
    logger.info("Starting cross-validation and error analysis")
    try:
        model_names = resolve_model_names(args.models)
        logger.event(
            "Step 4 config",
            model_scope="retained_model_families",
            cleanup_large_pretuning=args.cleanup_large_pretuning,
            save_full_scored=args.save_full_scored,
            train_fraction=train_fraction,
            overwrite=args.overwrite,
            resume=not args.overwrite,
            tfidf_profile=STEP4_REFERENCE_TFIDF["name"],
            selection_metric=config.selection_metric_label,
        )
        if not args.save_full_scored:
            logger.info("Compact step_4 mode is active; full OOF scored exports are disabled by default")
        if not args.overwrite:
            logger.info("Resume mode is active; completed units with matching step-4 profile will be skipped")
        if args.cleanup_large_pretuning:
            _cleanup_large_pretuning_artifacts(config, model_names, logger)
        else:
            logger.info("Skipping pretuning cleanup; use --cleanup-large-pretuning to remove old row-level artifacts")

        train, _, text_column = load_labeled_data(config)
        source_rows = len(train)
        if train_fraction < 1.0:
            train = sample_training_fraction(
                frame=train,
                target_cols=config.label_cols,
                train_fraction=train_fraction,
                random_seed=config.random_seed,
                n_splits=config.cv_folds,
            )
            logger.event(
                "Applied reproducible train downsampling",
                source_rows=source_rows,
                sampled_rows=len(train),
                train_fraction=train_fraction,
                stratification="multilabel-aware",
            )
        else:
            logger.info("Using the full train split for step 4")
        tfidf_overrides = step4_tfidf_overrides(text_column)
        logger.event(
            "Loaded train split",
            rows=len(train),
            source_rows=source_rows,
            text_column=text_column,
            models=", ".join(model_names),
        )

        for model_index, model_name in enumerate(model_names, start=1):
            logger.event(f"Model {model_index}/{len(model_names)} started", model_name=model_name)
            ml_root = approach_output_dir(config, model_name, "multilabel") / "pretuning_results"
            ml_signature = _build_run_signature(
                model_name=model_name,
                mode="multilabel",
                label_name=None,
                train_fraction=train_fraction,
                sampled_rows=len(train),
                source_rows=source_rows,
                text_column=text_column,
                save_full_scored=args.save_full_scored,
                config=config,
                tfidf_params={**STEP4_REFERENCE_TFIDF["params"], "lowercase": tfidf_overrides["tfidf__lowercase"], "dtype": "float32"},
            )
            ml_complete, ml_reason = _unit_is_complete(ml_root, ml_signature, args.save_full_scored)
            if ml_complete and not args.overwrite:
                logger.event("Skipping completed multilabel unit", indent=1, model_name=model_name)
                _remove_stale_full_scored_artifacts(ml_root, logger, indent=2)
            else:
                logger.info("Multilabel CV started", indent=1)
                if ml_reason != "ready":
                    logger.event("Multilabel unit will run", indent=2, reason=ml_reason)
                multilabel_cv = run_cv_experiment(
                    frame=train,
                    target_cols=config.label_cols,
                    text_column=text_column,
                    model_builder=lambda _params: build_multilabel_pipeline(model_name, text_column, config, tfidf_overrides),
                    model_name=model_name,
                    config=config,
                    logger=logger,
                    run_label=f"{model_name} multilabel baseline CV",
                    indent=1,
                    cleanup_memory=True,
                )
                _save_cv_outputs(
                    cv_result=multilabel_cv,
                    root=ml_root,
                    source_frame=train,
                    target_cols=config.label_cols,
                    text_column=text_column,
                    save_full_scored=args.save_full_scored,
                    logger=logger,
                    indent=1,
                )
                save_json(ml_signature, ml_root / RUN_CONFIG_FILENAME)
                logger.saved(ml_root / RUN_CONFIG_FILENAME, indent=1)
                ml_lookup = metric_lookup(multilabel_cv["oof_aggregate_metrics"])
                logger.event(
                    "Multilabel CV finished",
                    indent=1,
                    macro_pr_auc=ml_lookup.get("macro_pr_auc"),
                    macro_f1=ml_lookup.get("macro_f1"),
                    error_rows=len(multilabel_cv["error_samples"]),
                )
                del multilabel_cv
                gc.collect()

            for label_index, label_name in enumerate(config.label_cols, start=1):
                binary_root = approach_output_dir(config, model_name, "binary", label_name) / "pretuning_results"
                binary_signature = _build_run_signature(
                    model_name=model_name,
                    mode="binary",
                    label_name=label_name,
                    train_fraction=train_fraction,
                    sampled_rows=len(train),
                    source_rows=source_rows,
                    text_column=text_column,
                    save_full_scored=args.save_full_scored,
                    config=config,
                    tfidf_params={**STEP4_REFERENCE_TFIDF["params"], "lowercase": tfidf_overrides["tfidf__lowercase"], "dtype": "float32"},
                )
                binary_complete, binary_reason = _unit_is_complete(binary_root, binary_signature, args.save_full_scored)
                if binary_complete and not args.overwrite:
                    logger.event(
                        f"Binary CV {label_index}/{len(config.label_cols)} skipped",
                        indent=1,
                        label=label_name,
                    )
                    _remove_stale_full_scored_artifacts(binary_root, logger, indent=2)
                    continue

                logger.event(
                    f"Binary CV {label_index}/{len(config.label_cols)} started",
                    indent=1,
                    label=label_name,
                )
                if binary_reason != "ready":
                    logger.event("Binary unit will run", indent=2, label=label_name, reason=binary_reason)
                binary_cv = run_cv_experiment(
                    frame=train,
                    target_cols=[label_name],
                    text_column=text_column,
                    model_builder=lambda _params, label_name=label_name: build_binary_pipeline(model_name, text_column, config, tfidf_overrides),
                    model_name=model_name,
                    config=config,
                    logger=logger,
                    run_label=f"{model_name} binary baseline CV {label_name}",
                    indent=2,
                    cleanup_memory=True,
                )
                _save_cv_outputs(
                    cv_result=binary_cv,
                    root=binary_root,
                    source_frame=train,
                    target_cols=[label_name],
                    text_column=text_column,
                    save_full_scored=args.save_full_scored,
                    logger=logger,
                    indent=2,
                )
                save_json(binary_signature, binary_root / RUN_CONFIG_FILENAME)
                logger.saved(binary_root / RUN_CONFIG_FILENAME, indent=2)
                binary_lookup = metric_lookup(binary_cv["oof_aggregate_metrics"])
                logger.event(
                    "Binary CV finished",
                    indent=1,
                    label=label_name,
                    macro_pr_auc=binary_lookup.get("macro_pr_auc"),
                    macro_f1=binary_lookup.get("macro_f1"),
                    error_rows=len(binary_cv["error_samples"]),
                )
                del binary_cv
                gc.collect()
            logger.event("Model cross-validation outputs complete", model_name=model_name)
        logger.event("Cross-validation and error analysis complete", elapsed=logger.elapsed())
    finally:
        logger.close()


if __name__ == "__main__":
    main()
