from __future__ import annotations

import argparse
import time

from utils.config import build_config, ensure_project_dirs
from utils.io_utils import save_frame
from utils.metrics_utils import metric_value
from utils.metrics_utils import default_thresholds
from utils.modeling_utils import (
    build_binary_pipeline,
    build_multilabel_pipeline,
    evaluate_model,
    get_score_matrix,
    make_cv_splits,
)
from utils.pipeline_utils import add_model_args, approach_output_dir, load_labeled_data, resolve_model_names
from utils.progress_utils import build_step_logger, format_elapsed


def _first_holdout_indices(frame, target_cols, config):
    return make_cv_splits(frame[target_cols], random_seed=config.random_seed, n_splits=config.cv_folds)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train untuned baseline TF-IDF models.")
    add_model_args(parser)
    args = parser.parse_args()

    config = build_config()
    ensure_project_dirs(config)
    logger = build_step_logger(config, "step_3")
    logger.info("Starting baseline training")
    try:
        train, _, text_column = load_labeled_data(config)
        model_names = resolve_model_names(args.models)
        logger.event(
            "Loaded train split",
            rows=len(train),
            text_column=text_column,
            seed=config.random_seed,
            cv_folds=config.cv_folds,
        )
        logger.event("Model list ready", models=", ".join(model_names))

        ml_train_idx, ml_valid_idx = _first_holdout_indices(train, config.label_cols, config)
        multilabel_train = train.iloc[ml_train_idx].reset_index(drop=True)
        multilabel_valid = train.iloc[ml_valid_idx].reset_index(drop=True)
        binary_split_cache = {
            label_name: _first_holdout_indices(train, [label_name], config)
            for label_name in config.label_cols
        }
        logger.event(
            "Prepared baseline holdout splits",
            multilabel_train_rows=len(multilabel_train),
            multilabel_valid_rows=len(multilabel_valid),
        )

        for model_index, model_name in enumerate(model_names, start=1):
            model_start = time.perf_counter()
            logger.event(f"Model {model_index}/{len(model_names)} started", model_name=model_name)

            logger.info("Multilabel baseline started", indent=1)
            ml_run_start = time.perf_counter()
            multilabel_model = build_multilabel_pipeline(model_name, text_column, config)
            logger.info("Multilabel fit started", indent=2)
            multilabel_model.fit(multilabel_train[text_column], multilabel_train[config.label_cols])
            logger.info("Multilabel fit finished", indent=2)
            _, ml_score_kind = get_score_matrix(multilabel_model, multilabel_valid[text_column])
            multilabel_eval = evaluate_model(
                multilabel_model,
                multilabel_valid,
                config.label_cols,
                text_column,
                thresholds=default_thresholds(config.label_cols, ml_score_kind),
                logger=logger,
                run_label=f"{model_name} multilabel baseline",
                indent=2,
            )
            ml_root = approach_output_dir(config, model_name, "multilabel") / "pretuning_results"
            save_frame(multilabel_eval["label_metrics"], ml_root / "baseline_holdout_label_metrics.csv", index=False)
            save_frame(multilabel_eval["aggregate_metrics"], ml_root / "baseline_holdout_aggregate_metrics.csv", index=False)
            save_frame(multilabel_eval["error_samples"], ml_root / "baseline_holdout_error_samples.csv", index=False)
            logger.saved(ml_root / "baseline_holdout_label_metrics.csv", indent=2)
            logger.saved(ml_root / "baseline_holdout_aggregate_metrics.csv", indent=2)
            logger.saved(ml_root / "baseline_holdout_error_samples.csv", indent=2)
            logger.event(
                "Multilabel baseline finished",
                indent=1,
                macro_pr_auc=metric_value(multilabel_eval["aggregate_metrics"], "macro_pr_auc"),
                macro_f1=metric_value(multilabel_eval["aggregate_metrics"], "macro_f1"),
                elapsed=format_elapsed(time.perf_counter() - ml_run_start),
            )

            for label_index, label_name in enumerate(config.label_cols, start=1):
                label_start = time.perf_counter()
                logger.event(
                    f"Binary run {label_index}/{len(config.label_cols)} started",
                    indent=1,
                    label=label_name,
                )
                binary_train_idx, binary_valid_idx = binary_split_cache[label_name]
                binary_train = train.iloc[binary_train_idx].reset_index(drop=True)
                binary_valid = train.iloc[binary_valid_idx].reset_index(drop=True)
                binary_model = build_binary_pipeline(model_name, text_column, config)
                logger.event("Binary fit started", indent=2, label=label_name)
                binary_model.fit(binary_train[text_column], binary_train[label_name])
                logger.event("Binary fit finished", indent=2, label=label_name)
                _, binary_score_kind = get_score_matrix(binary_model, binary_valid[text_column])
                binary_eval = evaluate_model(
                    binary_model,
                    binary_valid,
                    [label_name],
                    text_column,
                    thresholds=default_thresholds([label_name], binary_score_kind),
                    logger=logger,
                    run_label=f"{model_name} binary baseline {label_name}",
                    indent=2,
                )
                binary_root = approach_output_dir(config, model_name, "binary", label_name) / "pretuning_results"
                save_frame(binary_eval["label_metrics"], binary_root / "baseline_holdout_label_metrics.csv", index=False)
                save_frame(binary_eval["aggregate_metrics"], binary_root / "baseline_holdout_aggregate_metrics.csv", index=False)
                save_frame(binary_eval["error_samples"], binary_root / "baseline_holdout_error_samples.csv", index=False)
                logger.saved(binary_root / "baseline_holdout_label_metrics.csv", indent=2)
                logger.saved(binary_root / "baseline_holdout_aggregate_metrics.csv", indent=2)
                logger.saved(binary_root / "baseline_holdout_error_samples.csv", indent=2)
                logger.event(
                    "Binary run finished",
                    indent=1,
                    label=label_name,
                    pr_auc=float(binary_eval["label_metrics"].iloc[0]["pr_auc"]),
                    f1=float(binary_eval["label_metrics"].iloc[0]["f1"]),
                    elapsed=format_elapsed(time.perf_counter() - label_start),
                )
            logger.event(
                "Model baseline outputs complete",
                model_name=model_name,
                elapsed=format_elapsed(time.perf_counter() - model_start),
            )
        logger.event("Baseline training complete", elapsed=logger.elapsed())
    finally:
        logger.close()


if __name__ == "__main__":
    main()
