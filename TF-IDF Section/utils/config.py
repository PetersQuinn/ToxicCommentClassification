from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TEXT_PRIORITY = [
    "comment_text_tfidf",
    "comment_text_clean",
    "comment_text",
]

DATA_FILES = {
    "train": "train_cleaned.parquet",
    "test": "test_cleaned.parquet",
    "test_labeled": "test_labeled_cleaned.parquet",
}

MODEL_FAMILIES = [
    "logistic_regression",
    "linear_svm",
    "complement_nb",
]

APPROACHES = ["multilabel", "binary"]

TFIDF_CANDIDATES = [
    {
        "name": "word_unigram_balanced",
        "params": {
            "analyzer": "word",
            "ngram_range": (1, 1),
            "min_df": 5,
            "max_df": 0.99,
            "sublinear_tf": False,
            "max_features": 75000,
            "norm": "l2",
            "lowercase": "auto",
        },
    },
    {
        "name": "word_bigram_balanced",
        "params": {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "min_df": 5,
            "max_df": 0.99,
            "sublinear_tf": True,
            "max_features": 120000,
            "norm": "l2",
            "lowercase": "auto",
        },
    },
    {
        "name": "word_bigram_high_vocab",
        "params": {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.99,
            "sublinear_tf": True,
            "max_features": 160000,
            "norm": "l2",
            "lowercase": "auto",
        },
    },
    {
        "name": "word_bigram_conservative",
        "params": {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "min_df": 10,
            "max_df": 0.95,
            "sublinear_tf": True,
            "max_features": 90000,
            "norm": "l2",
            "lowercase": "auto",
        },
    },
    {
        "name": "char_wb_3_5",
        "params": {
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "min_df": 5,
            "max_df": 1.0,
            "sublinear_tf": True,
            "max_features": 150000,
            "norm": "l2",
            "lowercase": True,
        },
    },
]

DEFAULT_REFERENCE_TFIDF = TFIDF_CANDIDATES[1]

STEP4_REFERENCE_TFIDF = {
    "name": "step4_word_bigram_light",
    "params": {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_df": 0.99,
        "sublinear_tf": True,
        "max_features": 50000,
        "norm": "l2",
        "lowercase": "auto",
    },
}

PROJECT_CONTEXT = (
    "This TF-IDF section covers the sparse lexical branch of our broader toxic "
    "comment classification project. Other sections of the repo handle the dense, "
    "transformer, hybrid, and fairness-specific analysis work."
)


@dataclass
class PipelineConfig:
    project_root: Path
    data_dir: Path
    outputs_dir: Path
    logs_dir: Path
    audit_dir: Path
    feature_dir: Path
    label_cols: list[str] = field(default_factory=lambda: list(LABEL_COLS))
    text_priority: list[str] = field(default_factory=lambda: list(TEXT_PRIORITY))
    random_seed: int = 701
    cv_folds: int = 5
    baseline_validation_fraction: float = 0.2
    core_search_iterations: int = 8
    model_parallel_jobs: int = 1
    max_error_examples_per_label: int = 30
    step4_default_train_fraction: float = 0.5
    selection_metric: str = "macro_pr_auc"
    selection_metric_label: str = "Macro Average Precision (macro PR-AUC)"
    selection_tie_break_metrics: list[str] = field(
        default_factory=lambda: ["micro_pr_auc", "macro_f1"]
    )


def _resolve_data_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "parquets",
        Path.home() / "data" / "parquets",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_config() -> PipelineConfig:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    return PipelineConfig(
        project_root=project_root,
        data_dir=_resolve_data_dir(project_root),
        outputs_dir=outputs_dir,
        logs_dir=outputs_dir / "logs",
        audit_dir=outputs_dir / "audit",
        feature_dir=outputs_dir / "feature_build",
    )


def model_dir(config: PipelineConfig, model_name: str) -> Path:
    return config.outputs_dir / "models" / model_name


def multilabel_dir(config: PipelineConfig, model_name: str) -> Path:
    return model_dir(config, model_name) / "multilabel"


def binary_dir(config: PipelineConfig, model_name: str, label_name: str | None = None) -> Path:
    base = model_dir(config, model_name) / "binary"
    return base if label_name is None else base / label_name


def comparison_dir(config: PipelineConfig, model_name: str) -> Path:
    return model_dir(config, model_name) / "comparison"


def ensure_project_dirs(config: PipelineConfig) -> None:
    base_dirs = [
        config.outputs_dir,
        config.logs_dir,
        config.audit_dir,
        config.feature_dir,
        config.outputs_dir / "models",
    ]
    for directory in base_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return value


def config_snapshot(config: PipelineConfig) -> dict[str, Any]:
    return to_serializable(asdict(config))
