"""Word-frequency and n-gram analysis helpers."""

import ast
from collections import Counter

import pandas as pd
from nltk import ngrams


def ensure_list_column(series: pd.Series) -> pd.Series:
    """Convert a string-serialized list column back to real lists."""
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])


def word_frequency(
    lemmas_series: pd.Series,
    top_n: int = 100,
) -> pd.DataFrame:
    """Return a DataFrame of the *top_n* most frequent lemmas."""
    all_lemmas = [lemma for lst in lemmas_series for lemma in lst]
    freq = Counter(all_lemmas)
    return pd.DataFrame(freq.most_common(top_n), columns=["word", "frequency"])


def compute_ngrams(
    lemmas_series: pd.Series,
    n: int = 2,
    min_freq: int = 5,
) -> pd.DataFrame:
    """Return a DataFrame of n-grams (joined with spaces) above *min_freq*."""
    all_ng: list[tuple] = []
    for lemma_list in lemmas_series:
        if len(lemma_list) >= n:
            all_ng.extend(list(ngrams(lemma_list, n)))

    freq = Counter(all_ng)
    rows = [
        (" ".join(ng), count)
        for ng, count in freq.items()
        if count >= min_freq
    ]
    col_name = "bigram" if n == 2 else "trigram" if n == 3 else f"{n}-gram"
    return pd.DataFrame(
        sorted(rows, key=lambda x: x[1], reverse=True),
        columns=[col_name, "frequency"],
    )
