from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.config import DEFAULT_REFERENCE_TFIDF, PipelineConfig, STEP4_REFERENCE_TFIDF
from utils.io_utils import ensure_dir
from utils.metrics_utils import (
    aggregate_metrics,
    apply_thresholds,
    default_thresholds,
    error_analysis_samples,
    metric_lookup,
    per_label_metrics,
    selection_values,
    tune_thresholds,
)
from utils.progress_utils import format_elapsed


@dataclass
class ModelSpec:
    estimator_factory: Callable[[int], Any]
    search_space: dict[str, list[Any]]
    tuning_sample_size: int | None = None
    search_iterations: int | None = None


def _binary_estimator(model_name: str, random_seed: int) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegression(C=1.0, class_weight=None, max_iter=2500, solver="liblinear")
    if model_name == "linear_svm":
        return LinearSVC(C=1.0, class_weight=None, dual="auto", max_iter=5000)
    if model_name == "complement_nb":
        return ComplementNB(alpha=1.0)
    raise KeyError(f"Unsupported model family: {model_name}")


def get_model_registry(config: PipelineConfig) -> dict[str, ModelSpec]:
    return {
        "logistic_regression": ModelSpec(
            estimator_factory=lambda seed: _binary_estimator("logistic_regression", seed),
            tuning_sample_size=100000,
            search_iterations=5,
            search_space={
                "tfidf__analyzer": ["word"],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [5, 10],
                "tfidf__max_df": [0.95, 0.99],
                "tfidf__sublinear_tf": [True, False],
                "tfidf__max_features": [50000, 80000, 120000],
                "clf__C": [0.5, 1.0, 2.0, 4.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "linear_svm": ModelSpec(
            estimator_factory=lambda seed: _binary_estimator("linear_svm", seed),
            tuning_sample_size=80000,
            search_iterations=5,
            search_space={
                "tfidf__analyzer": ["word"],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [5, 10],
                "tfidf__max_df": [0.95, 0.99],
                "tfidf__sublinear_tf": [True],
                "tfidf__max_features": [50000, 80000, 120000],
                "clf__C": [0.5, 1.0, 2.0, 4.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "complement_nb": ModelSpec(
            estimator_factory=lambda seed: _binary_estimator("complement_nb", seed),
            tuning_sample_size=100000,
            search_iterations=4,
            search_space={
                "tfidf__analyzer": ["word"],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [5, 10],
                "tfidf__max_df": [0.95, 0.99],
                "tfidf__sublinear_tf": [True, False],
                "tfidf__max_features": [50000, 80000, 120000],
                "clf__alpha": [0.1, 0.5, 1.0, 2.0],
            },
        ),
    }


def _auto_lowercase(text_column: str, analyzer: str) -> bool:
    if analyzer == "char_wb":
        return True
    return text_column != "comment_text_tfidf"


def _resolved_tfidf_params(text_column: str, profile: dict[str, Any]) -> dict[str, Any]:
    params = dict(profile["params"])
    params["lowercase"] = _auto_lowercase(text_column, params["analyzer"]) if params["lowercase"] == "auto" else bool(params["lowercase"])
    return params


def reference_tfidf_params(text_column: str) -> dict[str, Any]:
    return _resolved_tfidf_params(text_column, DEFAULT_REFERENCE_TFIDF)


def step4_tfidf_overrides(text_column: str) -> dict[str, Any]:
    params = _resolved_tfidf_params(text_column, STEP4_REFERENCE_TFIDF)
    overrides = {f"tfidf__{key}": value for key, value in params.items()}
    overrides["tfidf__dtype"] = np.float32
    return overrides


def _base_steps(model_name: str, text_column: str, config: PipelineConfig) -> list[tuple[str, Any]]:
    return [("tfidf", TfidfVectorizer(**reference_tfidf_params(text_column)))]


def build_binary_pipeline(
    model_name: str,
    text_column: str,
    config: PipelineConfig,
    overrides: dict[str, Any] | None = None,
) -> Pipeline:
    steps = _base_steps(model_name, text_column, config)
    steps.append(("clf", get_model_registry(config)[model_name].estimator_factory(config.random_seed)))
    model = Pipeline(steps)
    if overrides:
        model.set_params(**overrides)
        analyzer = model.get_params()["tfidf__analyzer"]
        model.set_params(tfidf__lowercase=_auto_lowercase(text_column, analyzer))
    return model


def build_multilabel_pipeline(
    model_name: str,
    text_column: str,
    config: PipelineConfig,
    overrides: dict[str, Any] | None = None,
) -> Pipeline:
    steps = _base_steps(model_name, text_column, config)
    estimator = get_model_registry(config)[model_name].estimator_factory(config.random_seed)
    steps.append(("clf", OneVsRestClassifier(estimator, n_jobs=config.model_parallel_jobs)))
    model = Pipeline(steps)
    if overrides:
        transformed = {}
        for key, value in overrides.items():
            if key.startswith("clf__"):
                transformed[key.replace("clf__", "clf__estimator__", 1)] = value
            else:
                transformed[key] = value
        model.set_params(**transformed)
        analyzer = model.get_params()["tfidf__analyzer"]
        model.set_params(tfidf__lowercase=_auto_lowercase(text_column, analyzer))
    return model


def get_score_matrix(model: Pipeline, text_values: pd.Series | list[str]) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        try:
            scores = np.asarray(model.predict_proba(text_values))
            if scores.ndim == 2 and scores.shape[1] == 2:
                scores = scores[:, 1].reshape(-1, 1)
            elif scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            return scores, "probability"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(text_values))
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            return scores, "margin"
        except Exception:
            pass
    scores = np.asarray(model.predict(text_values), dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores, "binary"


def multilabel_stratify_target(y_frame: pd.DataFrame, n_splits: int) -> pd.Series:
    bitmask = y_frame.astype(int).astype(str).agg("".join, axis=1)
    counts = bitmask.value_counts()
    target = bitmask.copy()
    positive_count = y_frame.sum(axis=1)
    fallback = []
    for row in y_frame.itertuples(index=False, name=None):
        labels = [y_frame.columns[idx] for idx, value in enumerate(row) if value == 1]
        if not labels:
            fallback.append("all_negative")
        elif len(labels) == 1:
            fallback.append(f"single::{labels[0]}")
        else:
            fallback.append("multi_positive")
    fallback = pd.Series(fallback, index=y_frame.index)
    rare = target.map(counts) < n_splits
    target.loc[rare] = fallback.loc[rare]
    still_rare = target.map(target.value_counts()) < n_splits
    target.loc[still_rare] = np.where(positive_count.loc[still_rare] > 0, "rare_positive", "rare_negative")
    if target.value_counts().min() < n_splits:
        target = pd.Series(np.where(positive_count > 0, "positive", "negative"), index=y_frame.index)
    return target


def make_cv_splits(y_frame: pd.DataFrame, random_seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if y_frame.shape[1] == 1:
        target = y_frame.iloc[:, 0].to_numpy()
    else:
        target = multilabel_stratify_target(y_frame, n_splits=n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    dummy_x = np.zeros(len(y_frame))
    return list(splitter.split(dummy_x, target))


def sample_training_subset(
    frame: pd.DataFrame,
    target_cols: list[str],
    max_rows: int | None,
    random_seed: int,
) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    y_frame = frame[target_cols]
    if len(target_cols) == 1:
        target = y_frame.iloc[:, 0]
    else:
        target = multilabel_stratify_target(y_frame, n_splits=5)
    sampled_idx, _ = train_test_split(
        np.arange(len(frame)),
        train_size=max_rows,
        random_state=random_seed,
        stratify=target,
    )
    return frame.iloc[np.sort(sampled_idx)].reset_index(drop=True)


def sample_training_fraction(
    frame: pd.DataFrame,
    target_cols: list[str],
    train_fraction: float,
    random_seed: int,
    n_splits: int = 5,
) -> pd.DataFrame:
    if train_fraction <= 0.0 or train_fraction > 1.0:
        raise ValueError("train_fraction must be in the interval (0, 1].")
    if train_fraction >= 1.0:
        return frame.reset_index(drop=True)

    sample_size = int(round(len(frame) * train_fraction))
    sample_size = max(sample_size, n_splits * 2)
    if sample_size >= len(frame):
        return frame.reset_index(drop=True)

    y_frame = frame[target_cols]
    if len(target_cols) == 1:
        target = y_frame.iloc[:, 0]
    else:
        target = multilabel_stratify_target(y_frame, n_splits=n_splits)

    sampled_idx, _ = train_test_split(
        np.arange(len(frame)),
        train_size=sample_size,
        random_state=random_seed,
        stratify=target,
    )
    return frame.iloc[np.sort(sampled_idx)].reset_index(drop=True)


def _selection_tuple(result_frame: pd.DataFrame, config: PipelineConfig) -> tuple[float, ...]:
    metric_names = [config.selection_metric] + list(config.selection_tie_break_metrics)
    return selection_values(result_frame, metric_names)


def _is_better_selection(
    candidate: tuple[float, ...],
    incumbent: tuple[float, ...] | None,
) -> bool:
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


def _aggregate_metric_rows(frame: pd.DataFrame, model_name: str, fold: int) -> list[dict[str, Any]]:
    copy = frame.copy()
    copy["model_name"] = model_name
    copy["fold"] = fold
    return copy.to_dict(orient="records")


def _label_metric_rows(frame: pd.DataFrame, model_name: str, fold: int) -> list[dict[str, Any]]:
    copy = frame.copy()
    copy["model_name"] = model_name
    copy["fold"] = fold
    return copy.to_dict(orient="records")


def run_cv_experiment(
    frame: pd.DataFrame,
    target_cols: list[str],
    text_column: str,
    model_builder: Callable[[dict[str, Any] | None], Pipeline],
    model_name: str,
    config: PipelineConfig,
    logger: Any | None = None,
    run_label: str | None = None,
    indent: int = 0,
    cleanup_memory: bool = False,
) -> dict[str, Any]:
    total_start: float | None = None
    splits = make_cv_splits(frame[target_cols], random_seed=config.random_seed, n_splits=config.cv_folds)
    oof_scores = np.zeros((len(frame), len(target_cols)), dtype=float)
    score_kinds: list[str] = []
    fold_rows: list[dict[str, Any]] = []
    fold_label_rows: list[dict[str, Any]] = []
    context_name = run_label or f"{model_name} CV"
    if logger is not None:
        logger.event(
            "Cross-validation started",
            indent=indent,
            context=context_name,
            folds=len(splits),
            rows=len(frame),
            targets=",".join(target_cols),
        )
        total_start = time.perf_counter()

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        valid_frame = frame.iloc[valid_idx].reset_index(drop=True)
        fold_start = time.perf_counter()
        if logger is not None:
            logger.event(
                f"Fold {fold_id}/{len(splits)} started",
                indent=indent + 1,
                train_rows=len(train_frame),
                valid_rows=len(valid_frame),
            )
        model = model_builder(None)
        y_train = train_frame[target_cols]
        if len(target_cols) == 1:
            y_train = y_train.iloc[:, 0]
        model.fit(train_frame[text_column], y_train)
        scores, score_kind = get_score_matrix(model, valid_frame[text_column])
        score_kinds.append(score_kind)
        oof_scores[valid_idx, :] = scores

        y_true = valid_frame[target_cols].reset_index(drop=True)
        thresholds = default_thresholds(target_cols, score_kind)
        y_pred = apply_thresholds(scores, thresholds, target_cols)
        fold_aggregate_metrics = aggregate_metrics(y_true, y_pred, scores, target_cols)
        fold_label_metrics = per_label_metrics(y_true, y_pred, scores, target_cols)
        fold_rows.extend(_aggregate_metric_rows(fold_aggregate_metrics, model_name, fold_id))
        fold_label_rows.extend(_label_metric_rows(fold_label_metrics, model_name, fold_id))
        if logger is not None:
            fold_lookup = metric_lookup(fold_aggregate_metrics)
            logger.event(
                f"Fold {fold_id}/{len(splits)} complete",
                indent=indent + 1,
                macro_pr_auc=fold_lookup.get("macro_pr_auc", np.nan),
                macro_f1=fold_lookup.get("macro_f1", np.nan),
                elapsed=format_elapsed(time.perf_counter() - fold_start),
            )
        if cleanup_memory:
            del model, train_frame, valid_frame, y_train, scores, y_true, y_pred, fold_aggregate_metrics, fold_label_metrics
            gc.collect()

    score_kind = "probability" if all(kind == "probability" for kind in score_kinds) else "margin"
    if logger is not None:
        logger.event(
            "Threshold tuning started",
            indent=indent,
            context=context_name,
            score_kind=score_kind,
        )
    best_thresholds, threshold_frame = tune_thresholds(frame[target_cols], oof_scores, target_cols, score_kind)
    oof_pred = apply_thresholds(oof_scores, best_thresholds, target_cols)
    result = {
        "fold_metrics": pd.DataFrame(fold_rows),
        "fold_label_metrics": pd.DataFrame(fold_label_rows),
        "oof_scores": oof_scores,
        "oof_pred": oof_pred,
        "score_kind": score_kind,
        "thresholds": best_thresholds,
        "threshold_frame": threshold_frame,
        "oof_label_metrics": per_label_metrics(frame[target_cols], oof_pred, oof_scores, target_cols),
        "oof_aggregate_metrics": aggregate_metrics(frame[target_cols], oof_pred, oof_scores, target_cols),
        "error_samples": error_analysis_samples(
            source_frame=frame.reset_index(drop=True),
            y_true=frame[target_cols].reset_index(drop=True),
            y_pred=oof_pred,
            score_matrix=oof_scores,
            thresholds=best_thresholds,
            label_cols=target_cols,
            text_col=text_column,
            top_n=config.max_error_examples_per_label,
        ),
    }
    if logger is not None and total_start is not None:
        oof_lookup = metric_lookup(result["oof_aggregate_metrics"])
        logger.event(
            "Cross-validation finished",
            indent=indent,
            context=context_name,
            macro_pr_auc=oof_lookup.get("macro_pr_auc", np.nan),
            macro_f1=oof_lookup.get("macro_f1", np.nan),
            elapsed=format_elapsed(time.perf_counter() - total_start),
        )
    if cleanup_memory:
        gc.collect()
    return result


def _n_iter_for_spec(spec: ModelSpec, config: PipelineConfig) -> int:
    if spec.search_iterations is not None:
        return spec.search_iterations
    return config.core_search_iterations


def run_search(
    frame: pd.DataFrame,
    target_cols: list[str],
    text_column: str,
    model_name: str,
    mode: str,
    config: PipelineConfig,
    logger: Any | None = None,
    run_label: str | None = None,
    indent: int = 0,
) -> dict[str, Any]:
    spec = get_model_registry(config)[model_name]
    sampled = sample_training_subset(frame, target_cols, spec.tuning_sample_size, config.random_seed)
    if mode == "multilabel":
        builder = lambda params: build_multilabel_pipeline(model_name, text_column, config, params)
    else:
        builder = lambda params: build_binary_pipeline(model_name, text_column, config, params)

    best_result: dict[str, Any] | None = None
    best_selection_tuple: tuple[float, ...] | None = None
    search_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_trials = _n_iter_for_spec(spec, config)
    context_name = run_label or f"{model_name} {mode}"
    search_start = time.perf_counter()
    if logger is not None:
        logger.event(
            "Search started",
            indent=indent,
            context=context_name,
            trials=total_trials,
            rows=len(sampled),
            selection_metric=config.selection_metric,
        )

    for trial_id, params in enumerate(
        ParameterSampler(spec.search_space, n_iter=total_trials, random_state=config.random_seed),
        start=1,
    ):
        trial_start = time.perf_counter()
        if logger is not None:
            logger.event(
                f"Trial {trial_id}/{total_trials} started",
                indent=indent + 1,
                context=context_name,
            )
        result = run_cv_experiment(
            frame=sampled,
            target_cols=target_cols,
            text_column=text_column,
            model_builder=lambda _: builder(params),
            model_name=model_name,
            config=config,
            logger=logger,
            run_label=f"{context_name} | trial {trial_id}",
            indent=indent + 2,
            cleanup_memory=True,
        )
        lookup = {row.metric: row.value for row in result["oof_aggregate_metrics"].itertuples(index=False)}
        selection_tuple = _selection_tuple(result["oof_aggregate_metrics"], config)
        primary_selection_score = selection_tuple[0]
        search_rows.append(
            {
                "trial_id": trial_id,
                "selection_metric": config.selection_metric,
                "selection_score": primary_selection_score,
                "params": params,
                **lookup,
            }
        )
        summary_rows.append(
            {
                "trial_id": trial_id,
                "selection_metric": config.selection_metric,
                "selection_score": primary_selection_score,
                "tfidf_name": f"{params.get('tfidf__analyzer')}__{params.get('tfidf__ngram_range')}",
                "params": params,
            }
        )
        if logger is not None:
            logger.event(
                f"Trial {trial_id}/{total_trials} complete",
                indent=indent + 1,
                selection_score=primary_selection_score,
                macro_pr_auc=lookup.get("macro_pr_auc", np.nan),
                macro_f1=lookup.get("macro_f1", np.nan),
                elapsed=format_elapsed(time.perf_counter() - trial_start),
            )
        if _is_better_selection(selection_tuple, best_selection_tuple):
            best_selection_tuple = selection_tuple
            best_result = {
                "best_params": params,
                "best_cv_result": result,
                "selection_score": primary_selection_score,
                "sampled_frame": sampled,
            }
        gc.collect()

    if best_result is None:
        raise RuntimeError(f"No search result produced for {model_name} ({mode})")
    if logger is not None:
        logger.event(
            "Search finished",
            indent=indent,
            context=context_name,
            best_selection_score=best_result["selection_score"],
            selection_metric=config.selection_metric,
            elapsed=format_elapsed(time.perf_counter() - search_start),
        )
    return {
        "search_frame": pd.DataFrame(search_rows).sort_values("selection_score", ascending=False),
        "tfidf_summary": pd.DataFrame(summary_rows).sort_values("selection_score", ascending=False),
        **best_result,
    }


def fit_final_model(
    frame: pd.DataFrame,
    target_cols: list[str],
    text_column: str,
    model_name: str,
    mode: str,
    config: PipelineConfig,
    best_params: dict[str, Any],
    logger: Any | None = None,
    run_label: str | None = None,
    indent: int = 0,
) -> Pipeline:
    fit_start = time.perf_counter()
    if logger is not None:
        logger.event(
            "Model fitting started",
            indent=indent,
            context=run_label or f"{model_name} {mode}",
            rows=len(frame),
            targets=",".join(target_cols),
        )
    if mode == "multilabel":
        model = build_multilabel_pipeline(model_name, text_column, config, best_params)
        y_train = frame[target_cols]
    else:
        model = build_binary_pipeline(model_name, text_column, config, best_params)
        y_train = frame[target_cols[0]]
    model.fit(frame[text_column], y_train)
    if logger is not None:
        logger.event(
            "Model fitting finished",
            indent=indent,
            context=run_label or f"{model_name} {mode}",
            elapsed=format_elapsed(time.perf_counter() - fit_start),
        )
    return model


def evaluate_model(
    model: Pipeline,
    frame: pd.DataFrame,
    target_cols: list[str],
    text_column: str,
    thresholds: dict[str, float],
    logger: Any | None = None,
    run_label: str | None = None,
    indent: int = 0,
) -> dict[str, Any]:
    eval_start = time.perf_counter()
    if logger is not None:
        logger.event(
            "Evaluation started",
            indent=indent,
            context=run_label or "evaluation",
            rows=len(frame),
        )
    y_true = frame[target_cols].reset_index(drop=True)
    scores, score_kind = get_score_matrix(model, frame[text_column])
    y_pred = apply_thresholds(scores, thresholds, target_cols)
    result = {
        "score_kind": score_kind,
        "scores": scores,
        "pred": y_pred,
        "label_metrics": per_label_metrics(y_true, y_pred, scores, target_cols),
        "aggregate_metrics": aggregate_metrics(y_true, y_pred, scores, target_cols),
        "error_samples": error_analysis_samples(
            source_frame=frame,
            y_true=y_true,
            y_pred=y_pred,
            score_matrix=scores,
            thresholds=thresholds,
            label_cols=target_cols,
            text_col=text_column,
            top_n=30,
        ),
    }
    if logger is not None:
        lookup = metric_lookup(result["aggregate_metrics"])
        logger.event(
            "Evaluation finished",
            indent=indent,
            context=run_label or "evaluation",
            macro_pr_auc=lookup.get("macro_pr_auc", np.nan),
            macro_f1=lookup.get("macro_f1", np.nan),
            elapsed=format_elapsed(time.perf_counter() - eval_start),
        )
    return result


def feature_attributions(model: Pipeline, target_cols: list[str]) -> pd.DataFrame:
    if "svd" in model.named_steps:
        return pd.DataFrame()
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    if not hasattr(vectorizer, "get_feature_names_out"):
        return pd.DataFrame()

    feature_names = vectorizer.get_feature_names_out()
    rows: list[dict[str, object]] = []
    estimators: list[Any]
    if hasattr(classifier, "estimators_"):
        estimators = list(classifier.estimators_)
    else:
        estimators = [classifier]

    for idx, label in enumerate(target_cols):
        estimator = estimators[idx]
        if hasattr(estimator, "coef_"):
            weights = np.asarray(estimator.coef_).ravel()
        elif hasattr(estimator, "feature_log_prob_") and estimator.feature_log_prob_.shape[0] >= 2:
            weights = np.asarray(estimator.feature_log_prob_[1] - estimator.feature_log_prob_[0]).ravel()
        else:
            continue
        pos_idx = np.argsort(-weights)[:20]
        neg_idx = np.argsort(weights)[:20]
        for rank, feature_idx in enumerate(pos_idx, start=1):
            rows.append({"label": label, "direction": "positive", "rank": rank, "feature": feature_names[feature_idx], "weight": float(weights[feature_idx])})
        for rank, feature_idx in enumerate(neg_idx, start=1):
            rows.append({"label": label, "direction": "negative", "rank": rank, "feature": feature_names[feature_idx], "weight": float(weights[feature_idx])})
    return pd.DataFrame(rows)


def scored_output_frame(
    source_frame: pd.DataFrame,
    scores: np.ndarray,
    pred: pd.DataFrame,
    thresholds: dict[str, float],
    target_cols: list[str],
    text_column: str,
) -> pd.DataFrame:
    frame = source_frame[["id", text_column]].copy().rename(columns={text_column: "text"})
    for idx, label in enumerate(target_cols):
        frame[f"{label}_score"] = scores[:, idx]
        frame[f"{label}_pred"] = pred[label].to_numpy()
        frame[f"{label}_threshold"] = thresholds[label]
        if label in source_frame.columns:
            frame[f"{label}_true"] = source_frame[label].to_numpy()
    return frame


def save_model(model: Pipeline, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(model, path)
