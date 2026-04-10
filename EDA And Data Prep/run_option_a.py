from __future__ import annotations

import pandas as pd

from src.data_io.load_data import load_datasets, schema_snapshot
from src.eda.run_eda import run_eda
from src.features.build_dense_features import build_dense_features, build_token_frequency
from src.features.build_tfidf_features import build_tfidf_matrices
from src.preprocessing.clean_text import apply_cleaning
from src.utils.config import get_config, set_global_seed
from src.utils.io_helpers import ensure_dirs, timestamp, write_dataframe, write_json, write_markdown
from src.utils.runtime import check_dependencies


def main() -> None:
    check_dependencies(["pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn", "pyarrow", "joblib"])
    config = get_config()
    set_global_seed(config.seed)
    ensure_dirs(config.output_dirs)
    config.save(config.output_dirs["tabular_metadata"] / "run_config.json")
    write_json(
        {
            "script": "run_option_a.py",
            "section": "data_exploration_and_tabular_prep",
            "timestamp": timestamp(),
        },
        config.output_dirs["tabular_metadata"] / "run_metadata.json",
    )

    print("Loading raw datasets...")
    bundle = load_datasets(config)
    train_df = apply_cleaning(bundle.train, config.text_column)
    test_df = apply_cleaning(bundle.test, config.text_column)

    print("Writing audit outputs...")
    write_json(schema_snapshot(bundle), config.output_dirs["audit"] / "schema_snapshot.json")
    audit_rows = []
    for name, df in {"train": train_df, "test": test_df, "test_labels": bundle.test_labels, "sample_submission": bundle.sample_submission}.items():
        for column in df.columns:
            audit_rows.append({"dataset": name, "column": column, "dtype": str(df[column].dtype), "null_count": int(df[column].isna().sum())})
    pd.DataFrame(audit_rows).to_csv(config.output_dirs["audit"] / "column_audit.csv", index=False)
    pd.DataFrame(
        [
            {"dataset": "train", "rows": len(train_df), "columns": train_df.shape[1]},
            {"dataset": "test", "rows": len(test_df), "columns": test_df.shape[1]},
            {"dataset": "test_labels", "rows": len(bundle.test_labels), "columns": bundle.test_labels.shape[1]},
            {"dataset": "sample_submission", "rows": len(bundle.sample_submission), "columns": bundle.sample_submission.shape[1]},
        ]
    ).to_csv(config.output_dirs["audit"] / "dataset_shapes.csv", index=False)
    pd.DataFrame([{"label": label, "positive_count": int(train_df[label].sum()), "prevalence": float(train_df[label].mean())} for label in config.labels]).to_csv(
        config.output_dirs["audit"] / "label_prevalence.csv",
        index=False,
    )
    test_label_usable = (bundle.test_labels[config.labels] != -1).all(axis=1)
    pd.DataFrame(
        [
            {"metric": "usable_test_label_rows", "value": int(test_label_usable.sum())},
            {"metric": "placeholder_test_label_rows", "value": int((~test_label_usable).sum())},
            {"metric": "train_null_comment_text", "value": int(bundle.train[config.text_column].isna().sum())},
            {"metric": "train_exact_duplicate_comment_text", "value": int(bundle.train[config.text_column].duplicated().sum())},
            {"metric": "train_max_char_length", "value": int(train_df["comment_text_clean"].str.len().max())},
            {"metric": "train_multilabel_rows", "value": int((train_df[config.labels].sum(axis=1) > 1).sum())},
        ]
    ).to_csv(config.output_dirs["audit"] / "audit_metrics.csv", index=False)

    write_markdown(
        """
# Audit Summary

- We expect `train.csv`, `test.csv`, `test_labels.csv`, and `sample_submission.csv` in `data/`.
- The train table matches the six-label Jigsaw format used by the rest of our project.
- `test_labels.csv` mixes usable rows with `-1` placeholders, so we keep it as a partial labeled reference instead of a standard split.
- Comment lengths are long-tailed, which matters for both EDA and later tabular feature design.
""",
        config.output_dirs["audit"] / "audit_summary.md",
    )

    print("Running EDA...")
    run_eda(train_df, config.output_dirs["figures_eda"], config.output_dirs["eda"], config)

    print("Building cleaned datasets...")
    usable_test_labels = bundle.test_labels.loc[test_label_usable].reset_index(drop=True)
    usable_test = test_df.merge(usable_test_labels, on="id", how="inner")
    write_dataframe(train_df, config.output_dirs["cleaned"] / "train_cleaned.parquet")
    write_dataframe(test_df, config.output_dirs["cleaned"] / "test_cleaned.parquet")
    write_dataframe(usable_test, config.output_dirs["cleaned"] / "test_labeled_cleaned.parquet")

    print("Building dense tabular features...")
    token_counter = build_token_frequency(train_df["comment_text_clean"])
    dense_train, subgroup_train = build_dense_features(train_df, token_counter)
    dense_test, subgroup_test = build_dense_features(test_df, token_counter)
    dense_train = train_df[["id", *config.labels]].merge(dense_train, on="id", how="left")
    dense_test = test_df[["id"]].merge(dense_test, on="id", how="left")
    write_dataframe(dense_train, config.output_dirs["tabular"] / "dense_engineered_train.parquet")
    write_dataframe(dense_test, config.output_dirs["tabular"] / "dense_engineered_test.parquet")
    write_dataframe(subgroup_train, config.output_dirs["tabular_diagnostics"] / "identity_diagnostic_train.parquet")
    write_dataframe(subgroup_test, config.output_dirs["tabular_diagnostics"] / "identity_diagnostic_test.parquet")
    subgroup_train.sum(numeric_only=True).reset_index(name="matched_rows").to_csv(
        config.output_dirs["tabular_diagnostics"] / "identity_group_counts_train.csv",
        index=False,
    )

    feature_dictionary = pd.DataFrame(
        [{"feature_name": col, "description": "Dense text feature exported for the tabular branch."} for col in dense_train.columns if col not in {"id", *config.labels}]
    )
    feature_dictionary.to_csv(config.output_dirs["tabular_metadata"] / "dense_feature_dictionary.csv", index=False)

    print("Building TF-IDF matrices...")
    train_tfidf, test_tfidf, _ = build_tfidf_matrices(
        train_df["comment_text_tfidf"],
        test_df["comment_text_tfidf"],
        config,
        config.output_dirs["tabular_matrices"],
    )
    pd.DataFrame({"row_index": range(len(train_df)), "id": train_df["id"]}).to_csv(config.output_dirs["tabular_metadata"] / "train_row_alignment.csv", index=False)
    pd.DataFrame({"row_index": range(len(test_df)), "id": test_df["id"]}).to_csv(config.output_dirs["tabular_metadata"] / "test_row_alignment.csv", index=False)
    write_json(
        {
            "tfidf_shape_train": [int(train_tfidf.shape[0]), int(train_tfidf.shape[1])],
            "tfidf_shape_test": [int(test_tfidf.shape[0]), int(test_tfidf.shape[1])],
            "tfidf_params": {"ngram_range": [1, 2], "min_df": config.tfidf_min_df, "max_df": config.tfidf_max_df, "max_features": config.tfidf_max_features, "sublinear_tf": True},
        },
        config.output_dirs["tabular_metadata"] / "tfidf_metadata.json",
    )
    print("Data exploration and tabular preprocessing complete.")


if __name__ == "__main__":
    main()
