from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import PipelineConfig


def build_tfidf_matrices(
    train_text: pd.Series,
    test_text: pd.Series,
    config: PipelineConfig,
    output_dir: Path,
) -> Tuple[csr_matrix, csr_matrix, pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=config.tfidf_min_df,
        max_df=config.tfidf_max_df,
        max_features=config.tfidf_max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    save_npz(output_dir / "train_tfidf.npz", train_matrix)
    save_npz(output_dir / "test_tfidf.npz", test_matrix)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    feature_map = pd.DataFrame(
        {
            "feature_index": range(len(vectorizer.get_feature_names_out())),
            "feature_name": vectorizer.get_feature_names_out(),
        }
    )
    feature_map.to_csv(output_dir / "tfidf_feature_map.csv", index=False)
    return train_matrix, test_matrix, feature_map
