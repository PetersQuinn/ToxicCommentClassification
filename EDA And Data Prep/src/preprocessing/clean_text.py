from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTISPACE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = CONTROL_CHARS.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_for_classical(text: str) -> str:
    return normalize_text(text)


def clean_for_tfidf(text: str) -> str:
    return clean_for_classical(text).lower()


def apply_cleaning(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    result = df.copy()
    result["comment_text_raw"] = result[text_column].fillna("")
    result["comment_text_clean"] = result["comment_text_raw"].map(clean_for_classical)
    result["comment_text_tfidf"] = result["comment_text_raw"].map(clean_for_tfidf)
    return result


def build_length_buckets(series: Iterable[int]) -> pd.Categorical:
    bins = [0, 50, 100, 200, 400, 800, 1600, 5001]
    labels = ["0-50", "51-100", "101-200", "201-400", "401-800", "801-1600", "1601+"]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=True)
