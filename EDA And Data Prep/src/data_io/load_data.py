from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.utils.config import PipelineConfig


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    test_labels: pd.DataFrame
    sample_submission: pd.DataFrame


def _validate_columns(df: pd.DataFrame, expected: List[str], name: str) -> None:
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing expected columns: {missing}")


def load_datasets(config: PipelineConfig) -> DatasetBundle:
    required_paths = [
        config.train_path,
        config.test_path,
        config.test_labels_path,
        config.sample_submission_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset files: {missing}")

    train = pd.read_csv(config.train_path)
    test = pd.read_csv(config.test_path)
    test_labels = pd.read_csv(config.test_labels_path)
    sample_submission = pd.read_csv(config.sample_submission_path)

    _validate_columns(train, [config.id_column, config.text_column, *config.labels], "train.csv")
    _validate_columns(test, [config.id_column, config.text_column], "test.csv")
    _validate_columns(test_labels, [config.id_column, *config.labels], "test_labels.csv")
    _validate_columns(sample_submission, [config.id_column, *config.labels], "sample_submission.csv")
    return DatasetBundle(train=train, test=test, test_labels=test_labels, sample_submission=sample_submission)


def schema_snapshot(bundle: DatasetBundle) -> Dict[str, Dict[str, object]]:
    snapshot: Dict[str, Dict[str, object]] = {}
    for name, df in {
        "train": bundle.train,
        "test": bundle.test,
        "test_labels": bundle.test_labels,
        "sample_submission": bundle.sample_submission,
    }.items():
        snapshot[name] = {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
    return snapshot
