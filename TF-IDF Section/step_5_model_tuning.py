from __future__ import annotations

import argparse

from utils.config import build_config, ensure_project_dirs
from utils.pipeline_utils import (
    add_model_args,
    approach_output_dir,
    load_labeled_data,
    resolve_model_names,
    save_search_bundle,
)
from utils.modeling_utils import run_search
from utils.progress_utils import build_step_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune multilabel and binary TF-IDF models.")
    add_model_args(parser)
    args = parser.parse_args()

    config = build_config()
    ensure_project_dirs(config)
    logger = build_step_logger(config, "step_5")
    logger.info("Starting model tuning")
    try:
        train, _, text_column = load_labeled_data(config)
        model_names = resolve_model_names(args.models)
        logger.event(
            "Loaded train split",
            rows=len(train),
            text_column=text_column,
            models=", ".join(model_names),
            selection_metric=config.selection_metric_label,
        )

        for model_index, model_name in enumerate(model_names, start=1):
            logger.event(f"Model {model_index}/{len(model_names)} tuning started", model_name=model_name)
            logger.info("Multilabel tuning started", indent=1)
            multilabel_search = run_search(
                frame=train,
                target_cols=config.label_cols,
                text_column=text_column,
                model_name=model_name,
                mode="multilabel",
                config=config,
                logger=logger,
                run_label=f"{model_name} multilabel tuning",
                indent=1,
            )
            save_search_bundle(
                root=approach_output_dir(config, model_name, "multilabel"),
                best_params=multilabel_search["best_params"],
                search_frame=multilabel_search["search_frame"],
                tfidf_summary=multilabel_search["tfidf_summary"],
                cv_result=multilabel_search["best_cv_result"],
                logger=logger,
            )

            for label_index, label_name in enumerate(config.label_cols, start=1):
                logger.event(
                    f"Binary tuning {label_index}/{len(config.label_cols)} started",
                    indent=1,
                    label=label_name,
                )
                binary_search = run_search(
                    frame=train,
                    target_cols=[label_name],
                    text_column=text_column,
                    model_name=model_name,
                    mode="binary",
                    config=config,
                    logger=logger,
                    run_label=f"{model_name} binary tuning {label_name}",
                    indent=2,
                )
                save_search_bundle(
                    root=approach_output_dir(config, model_name, "binary", label_name),
                    best_params=binary_search["best_params"],
                    search_frame=binary_search["search_frame"],
                    tfidf_summary=binary_search["tfidf_summary"],
                    cv_result=binary_search["best_cv_result"],
                    logger=logger,
                )
            logger.event("Model tuning outputs complete", model_name=model_name)
        logger.event("Model tuning complete", elapsed=logger.elapsed(), selection_metric=config.selection_metric_label)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
