from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, Iterable, Tuple

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.features.lexicons import IDENTITY_GROUPS, NEGATIVE_SENTIMENT, POSITIVE_SENTIMENT, PROFANITY_TERMS

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#\w+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
ELONGATED_PATTERN = re.compile(r"(.)\1{2,}", re.IGNORECASE)
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,])\1{1,}")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def count_group_terms(tokens: Iterable[str], lexicon: Iterable[str]) -> int:
    lex = set(lexicon)
    return sum(token in lex for token in tokens)


def build_token_frequency(texts: pd.Series) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return counts


def build_dense_features(df: pd.DataFrame, rare_token_counter: Counter, text_col: str = "comment_text_clean") -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    subgroup_records = []
    punctuation = {mark: re.compile(re.escape(mark)) for mark in ["!", "?", ".", ",", "\"", "'"]}
    stopwords = set(ENGLISH_STOP_WORDS)

    for row in df[[text_col, "comment_text_raw", "id"]].itertuples(index=False):
        text = getattr(row, text_col)
        raw_text = row.comment_text_raw
        tokens = tokenize(text)
        n_chars = len(text)
        n_words = len(tokens)
        n_sentences = max(1, len(re.findall(r"[.!?]+", text))) if text else 0
        uppercase_chars = sum(char.isupper() for char in raw_text)
        digits = sum(char.isdigit() for char in text)
        whitespace = sum(char.isspace() for char in text)
        non_alnum = sum(not char.isalnum() and not char.isspace() for char in text)
        unique_tokens = len(set(tokens))
        stopword_count = sum(token in stopwords for token in tokens)
        profanity_count = count_group_terms(tokens, PROFANITY_TERMS)
        positive_count = count_group_terms(tokens, POSITIVE_SENTIMENT)
        negative_count = count_group_terms(tokens, NEGATIVE_SENTIMENT)
        rare_count = sum(rare_token_counter.get(token, 0) <= 2 for token in tokens)
        title_case_words = sum(word.istitle() for word in raw_text.split())
        all_caps_words = sum(word.isupper() and len(word) > 1 for word in raw_text.split())
        subgroup_row: Dict[str, int] = {"id": row.id}
        for name, group_terms in IDENTITY_GROUPS.items():
            subgroup_row[f"identity_{name}"] = int(any(token in group_terms for token in tokens))
        subgroup_records.append(subgroup_row)

        record = {
            "id": row.id,
            "char_count": n_chars,
            "word_count": n_words,
            "sentence_count": n_sentences,
            "avg_word_length": (sum(len(token) for token in tokens) / n_words) if n_words else 0.0,
            "avg_sentence_length_words": (n_words / n_sentences) if n_sentences else 0.0,
            "lexical_diversity": (unique_tokens / n_words) if n_words else 0.0,
            "hapax_ratio": (sum(count == 1 for count in Counter(tokens).values()) / n_words) if n_words else 0.0,
            "uppercase_count": uppercase_chars,
            "uppercase_ratio": (uppercase_chars / max(1, len(raw_text))),
            "digit_count": digits,
            "digit_ratio": digits / max(1, n_chars),
            "whitespace_count": whitespace,
            "whitespace_ratio": whitespace / max(1, n_chars),
            "non_alnum_ratio": non_alnum / max(1, n_chars),
            "url_count": len(URL_PATTERN.findall(raw_text)),
            "email_count": len(EMAIL_PATTERN.findall(raw_text)),
            "mention_count": len(MENTION_PATTERN.findall(raw_text)),
            "hashtag_count": len(HASHTAG_PATTERN.findall(raw_text)),
            "elongated_word_count": len(ELONGATED_PATTERN.findall(raw_text)),
            "repeated_punct_count": len(REPEATED_PUNCT_PATTERN.findall(raw_text)),
            "all_caps_word_count": all_caps_words,
            "title_case_word_ratio": title_case_words / max(1, len(raw_text.split())),
            "stopword_ratio": stopword_count / max(1, n_words),
            "profanity_count": profanity_count,
            "profanity_ratio": profanity_count / max(1, n_words),
            "positive_lexicon_count": positive_count,
            "negative_lexicon_count": negative_count,
            "sentiment_lexicon_balance": (positive_count - negative_count) / max(1, n_words),
            "rare_token_ratio": rare_count / max(1, n_words),
            "char_entropy": char_entropy(raw_text),
            "emoji_like_count": sum(char not in string.printable and not char.isspace() for char in raw_text),
            "quote_count": raw_text.count('"') + raw_text.count("'"),
        }
        for symbol, pattern in punctuation.items():
            key = {
                "!": "exclamation_count",
                "?": "question_count",
                ".": "period_count",
                ",": "comma_count",
                "\"": "double_quote_count",
                "'": "single_quote_count",
            }[symbol]
            record[key] = len(pattern.findall(raw_text))
        records.append(record)

    return pd.DataFrame(records), pd.DataFrame(subgroup_records)
