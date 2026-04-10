import pandas as pd

from src.preprocessing.clean_text import apply_cleaning, clean_for_classical


def test_clean_for_classical_is_deterministic():
    text = "Hello\r\nWorld!!!"
    assert clean_for_classical(text) == "Hello\nWorld!!!"


def test_apply_cleaning_adds_only_tabular_text_columns():
    df = pd.DataFrame({"comment_text": ["Hi THERE"]})

    cleaned = apply_cleaning(df, "comment_text")

    assert {"comment_text_raw", "comment_text_clean", "comment_text_tfidf"}.issubset(cleaned.columns)
    assert "comment_text_transformer" not in cleaned.columns
    assert cleaned.loc[0, "comment_text_tfidf"] == "hi there"
