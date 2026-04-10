from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class PipelineConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    train_path: Path = field(init=False)
    test_path: Path = field(init=False)
    test_labels_path: Path = field(init=False)
    sample_submission_path: Path = field(init=False)
    labels: List[str] = field(
        default_factory=lambda: [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
    )
    text_column: str = "comment_text"
    id_column: str = "id"
    seed: int = 705
    tfidf_max_features: int = 50000
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.95

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.outputs_dir = self.project_root / "outputs"
        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"
        self.test_labels_path = self.data_dir / "test_labels.csv"
        self.sample_submission_path = self.data_dir / "sample_submission.csv"

    @property
    def output_dirs(self) -> Dict[str, Path]:
        return {
            "audit": self.outputs_dir / "01_data_audit",
            "eda": self.outputs_dir / "02_eda",
            "cleaned": self.outputs_dir / "03_cleaned_data",
            "tabular": self.outputs_dir / "04_option_a_tabular",
            "tabular_matrices": self.outputs_dir / "04_option_a_tabular" / "matrices",
            "tabular_metadata": self.outputs_dir / "04_option_a_tabular" / "metadata",
            "tabular_diagnostics": self.outputs_dir / "04_option_a_tabular" / "diagnostics",
            "figures": self.outputs_dir / "07_figures",
            "figures_eda": self.outputs_dir / "07_figures" / "eda",
        }

    def to_jsonable(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["data_dir"] = str(self.data_dir)
        payload["outputs_dir"] = str(self.outputs_dir)
        payload["train_path"] = str(self.train_path)
        payload["test_path"] = str(self.test_path)
        payload["test_labels_path"] = str(self.test_labels_path)
        payload["sample_submission_path"] = str(self.sample_submission_path)
        return payload

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_jsonable(), indent=2), encoding="utf-8")


def get_config() -> PipelineConfig:
    return PipelineConfig()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
