from __future__ import annotations

import pandas as pd

from utils.config import build_config, ensure_project_dirs
from utils.io_utils import choose_text_column, load_parquet_frame, save_frame
from utils.plotting_utils import plot_bar, plot_histogram
from utils.progress_utils import build_step_logger


def _schema_rows(frame: pd.DataFrame, split_name: str) -> list[dict[str, object]]:
    return [
        {
            "split": split_name,
            "column": column,
            "dtype": str(frame[column].dtype),
            "null_count": int(frame[column].isna().sum()),
        }
        for column in frame.columns
    ]


def _text_length_summary(frame: pd.DataFrame, split_name: str, text_column: str) -> pd.DataFrame:
    text = frame[text_column].fillna("").astype(str)
    word_length = text.str.split().str.len()
    char_length = text.str.len()
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    return pd.DataFrame(
        {
            "split": split_name,
            "measure": ["word_count", "char_count"],
            "mean": [float(word_length.mean()), float(char_length.mean())],
            "median": [float(word_length.median()), float(char_length.median())],
            "q0": [float(word_length.quantile(quantiles[0])), float(char_length.quantile(quantiles[0]))],
            "q25": [float(word_length.quantile(quantiles[1])), float(char_length.quantile(quantiles[1]))],
            "q50": [float(word_length.quantile(quantiles[2])), float(char_length.quantile(quantiles[2]))],
            "q75": [float(word_length.quantile(quantiles[3])), float(char_length.quantile(quantiles[3]))],
            "q90": [float(word_length.quantile(quantiles[4])), float(char_length.quantile(quantiles[4]))],
            "q99": [float(word_length.quantile(quantiles[5])), float(char_length.quantile(quantiles[5]))],
            "q100": [float(word_length.quantile(quantiles[6])), float(char_length.quantile(quantiles[6]))],
        }
    )


def main() -> None:
    config = build_config()
    ensure_project_dirs(config)
    logger = build_step_logger(config, "step_1")
    logger.info("Starting data audit and validation")

    try:
        train = load_parquet_frame(config, "train")
        test = load_parquet_frame(config, "test")
        test_labeled = load_parquet_frame(config, "test_labeled")
        text_column = choose_text_column(train, config.text_priority)
        logger.event(
            "Loaded parquet splits",
            train_rows=len(train),
            test_rows=len(test),
            test_labeled_rows=len(test_labeled),
            text_column=text_column,
        )

        shape_frame = pd.DataFrame(
            [
                {"split": "train", "rows": len(train), "columns": train.shape[1]},
                {"split": "test", "rows": len(test), "columns": test.shape[1]},
                {"split": "test_labeled", "rows": len(test_labeled), "columns": test_labeled.shape[1]},
            ]
        )
        duplicate_frame = pd.DataFrame(
            [
                {"split": "train", "duplicate_ids": int(train["id"].duplicated().sum())},
                {"split": "test", "duplicate_ids": int(test["id"].duplicated().sum())},
                {"split": "test_labeled", "duplicate_ids": int(test_labeled["id"].duplicated().sum())},
            ]
        )
        schema_frame = pd.DataFrame(
            _schema_rows(train, "train")
            + _schema_rows(test, "test")
            + _schema_rows(test_labeled, "test_labeled")
        )
        null_frame = schema_frame[["split", "column", "null_count"]].copy()

        prevalence_rows = []
        for split_name, frame in [("train", train), ("test_labeled", test_labeled)]:
            for label_name in config.label_cols:
                prevalence_rows.append(
                    {
                        "split": split_name,
                        "label": label_name,
                        "positive_count": int(frame[label_name].sum()),
                        "prevalence": float(frame[label_name].mean()),
                    }
                )
        prevalence_frame = pd.DataFrame(prevalence_rows)

        text_length_frame = pd.concat(
            [
                _text_length_summary(train, "train", text_column),
                _text_length_summary(test_labeled, "test_labeled", text_column),
            ],
            ignore_index=True,
        )

        save_frame(shape_frame, config.audit_dir / "dataset_shapes.csv", index=False)
        save_frame(duplicate_frame, config.audit_dir / "duplicate_summary.csv", index=False)
        save_frame(schema_frame, config.audit_dir / "schema_summary.csv", index=False)
        save_frame(null_frame, config.audit_dir / "null_summary.csv", index=False)
        save_frame(prevalence_frame, config.audit_dir / "label_prevalence.csv", index=False)
        save_frame(text_length_frame, config.audit_dir / "text_length_summary.csv", index=False)
        logger.saved(config.audit_dir / "dataset_shapes.csv")
        logger.saved(config.audit_dir / "duplicate_summary.csv")
        logger.saved(config.audit_dir / "schema_summary.csv")
        logger.saved(config.audit_dir / "null_summary.csv")
        logger.saved(config.audit_dir / "label_prevalence.csv")
        logger.saved(config.audit_dir / "text_length_summary.csv")

        plot_bar(
            prevalence_frame[prevalence_frame["split"] == "train"],
            x_col="label",
            y_col="prevalence",
            title="Train Label Prevalence",
            ylabel="Prevalence",
            path=config.audit_dir / "train_label_prevalence.png",
        )
        plot_histogram(
            train[text_column].fillna("").astype(str).str.split().str.len(),
            title=f"Train Word Count Distribution ({text_column})",
            xlabel="Word count",
            path=config.audit_dir / "train_word_count_histogram.png",
            bins=60,
        )
        plot_histogram(
            test_labeled[text_column].fillna("").astype(str).str.split().str.len(),
            title=f"Test Word Count Distribution ({text_column})",
            xlabel="Word count",
            path=config.audit_dir / "test_word_count_histogram.png",
            bins=60,
        )
        logger.saved(config.audit_dir / "train_label_prevalence.png")
        logger.saved(config.audit_dir / "train_word_count_histogram.png")
        logger.saved(config.audit_dir / "test_word_count_histogram.png")
        logger.event("Data audit complete", elapsed=logger.elapsed())
    finally:
        logger.close()


if __name__ == "__main__":
    main()
