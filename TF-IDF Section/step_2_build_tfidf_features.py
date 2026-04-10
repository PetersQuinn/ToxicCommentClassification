from __future__ import annotations

import joblib
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.config import TFIDF_CANDIDATES, build_config, ensure_project_dirs
from utils.io_utils import save_frame, save_json
from utils.pipeline_utils import load_labeled_data
from utils.progress_utils import build_step_logger, format_elapsed


def _resolve_lowercase(text_column: str, analyzer: str, configured_value) -> bool:
    if configured_value != "auto":
        return bool(configured_value)
    if analyzer == "char_wb":
        return True
    return text_column != "comment_text_tfidf"


def main() -> None:
    config = build_config()
    ensure_project_dirs(config)
    logger = build_step_logger(config, "step_2")
    logger.info("Starting TF-IDF feature build")
    try:
        train, test_labeled, text_column = load_labeled_data(config)
        logger.event(
            "Loaded labeled data",
            train_rows=len(train),
            test_labeled_rows=len(test_labeled),
            text_column=text_column,
        )

        candidate_rows = []
        best_vectorizer = None
        best_name = None
        best_feature_count = -1
        for index, candidate in enumerate(TFIDF_CANDIDATES, start=1):
            params = dict(candidate["params"])
            params["lowercase"] = _resolve_lowercase(text_column, params["analyzer"], params["lowercase"])
            logger.event(
                f"Candidate {index}/{len(TFIDF_CANDIDATES)} started",
                candidate_name=candidate["name"],
                analyzer=params["analyzer"],
                ngram_range=params["ngram_range"],
            )
            start = time.perf_counter()
            vectorizer = TfidfVectorizer(**params)
            train_matrix = vectorizer.fit_transform(train[text_column])
            test_matrix = vectorizer.transform(test_labeled[text_column])
            feature_count = len(vectorizer.get_feature_names_out())
            candidate_rows.append(
                {
                    "candidate_name": candidate["name"],
                    "analyzer": params["analyzer"],
                    "ngram_range": str(params["ngram_range"]),
                    "min_df": params["min_df"],
                    "max_df": params["max_df"],
                    "sublinear_tf": params["sublinear_tf"],
                    "max_features": params["max_features"],
                    "lowercase": params["lowercase"],
                    "train_shape": str(train_matrix.shape),
                    "test_shape": str(test_matrix.shape),
                    "feature_count": feature_count,
                    "train_density": float(train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])),
                    "test_density": float(test_matrix.nnz / (test_matrix.shape[0] * test_matrix.shape[1])),
                }
            )
            if feature_count > best_feature_count:
                best_feature_count = feature_count
                best_vectorizer = vectorizer
                best_name = candidate["name"]
            logger.event(
                f"Candidate {index}/{len(TFIDF_CANDIDATES)} finished",
                candidate_name=candidate["name"],
                feature_count=feature_count,
                elapsed=format_elapsed(time.perf_counter() - start),
            )

        candidate_frame = pd.DataFrame(candidate_rows).sort_values("feature_count", ascending=False)

        save_frame(candidate_frame, config.feature_dir / "tfidf_candidate_summary.csv", index=False)
        save_json(
            {
                "text_column": text_column,
                "reference_vectorizer_name": best_name,
                "candidate_names": [candidate["name"] for candidate in TFIDF_CANDIDATES],
            },
            config.feature_dir / "tfidf_build_metadata.json",
        )
        if best_vectorizer is not None:
            joblib.dump(best_vectorizer, config.feature_dir / "reference_tfidf_vectorizer.joblib")
        logger.saved(config.feature_dir / "tfidf_candidate_summary.csv")
        logger.saved(config.feature_dir / "tfidf_build_metadata.json")
        if best_vectorizer is not None:
            logger.saved(config.feature_dir / "reference_tfidf_vectorizer.joblib")
        logger.event("TF-IDF feature build complete", reference_vectorizer=best_name, elapsed=logger.elapsed())
    finally:
        logger.close()


if __name__ == "__main__":
    main()
