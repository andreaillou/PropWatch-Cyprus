"""Russian text lemmatization using spaCy."""

import pandas as pd


def lemmatize_russian(text, nlp) -> list[str]:
    """Lemmatize a single Russian text string.

    Parameters
    ----------
    text : str or NaN
        Input text.
    nlp : spacy.Language
        A loaded spaCy Russian model (e.g. ``ru_core_news_lg``).

    Returns
    -------
    list[str]
        Filtered lemmas (lowercase, alphabetic, length > 2, no stop-words).
    """
    if pd.isna(text) or text == "":
        return []

    doc = nlp(str(text))
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
        and len(token.text) > 2
    ]


def lemmatize_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
    nlp=None,
) -> pd.DataFrame:
    """Add a ``lemmas`` column to *df*.

    If *nlp* is ``None`` the function will attempt to load
    ``ru_core_news_lg`` (requires it to be installed).
    """
    if nlp is None:
        import spacy

        nlp = spacy.load("ru_core_news_lg")

    print("Starting Russian lemmatization with spaCy...")
    df["lemmas"] = df[text_col].apply(lambda t: lemmatize_russian(t, nlp))
    print(f"Completed! Processed {len(df)} Russian posts")
    return df
