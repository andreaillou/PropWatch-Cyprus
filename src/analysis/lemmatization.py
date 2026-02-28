"""Multilingual lemmatization using spaCy (Russian / Cyrillic, Greek, Latin/English)."""

import pandas as pd


# ── Shared core ──────────────────────────────────────────────────────────────────

def _lemmatize(text, nlp) -> list[str]:
    """Shared lemmatization logic for any spaCy model."""
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


# ── Per-language convenience wrappers ────────────────────────────────────────────

def lemmatize_russian(text, nlp) -> list[str]:
    """Lemmatize a single Russian text string.

    Parameters
    ----------
    text : str or NaN
    nlp : spacy.Language
        A loaded spaCy Russian model (e.g. ``ru_core_news_lg``).

    Returns
    -------
    list[str]
        Filtered lemmas (lowercase, alphabetic, length > 2, no stop-words).
    """
    return _lemmatize(text, nlp)


def lemmatize_greek(text, nlp) -> list[str]:
    """Lemmatize a single Modern Greek text string.

    Parameters
    ----------
    text : str or NaN
    nlp : spacy.Language
        A loaded spaCy Greek model (e.g. ``el_core_news_lg``).

    Returns
    -------
    list[str]
        Filtered lemmas (lowercase, alphabetic, length > 2, no stop-words).
    """
    return _lemmatize(text, nlp)


def lemmatize_latin(text, nlp) -> list[str]:
    """Lemmatize a single Latin-script (English / other) text string.

    Parameters
    ----------
    text : str or NaN
    nlp : spacy.Language
        A loaded spaCy Latin-script model (e.g. ``en_core_web_lg``).

    Returns
    -------
    list[str]
        Filtered lemmas (lowercase, alphabetic, length > 2, no stop-words).
    """
    return _lemmatize(text, nlp)


# ── Column-level helper ───────────────────────────────────────────────────────────

_SCRIPT_MODEL_DEFAULTS: dict[str, str] = {
    "cyrillic": "ru_core_news_lg",
    "greek":    "el_core_news_lg",
    "latin":    "en_core_web_lg",
}


def lemmatize_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
    script: str = "cyrillic",
    nlp=None,
) -> pd.DataFrame:
    """Add a ``lemmas`` column to *df* using the appropriate spaCy model.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Column containing cleaned text.
    script : ``'cyrillic'`` | ``'greek'`` | ``'latin'``
        Script type; determines which spaCy model to load automatically when
        *nlp* is ``None``.

        * ``'cyrillic'`` → ``ru_core_news_lg``
        * ``'greek'``    → ``el_core_news_lg``
        * ``'latin'``    → ``en_core_web_lg``
    nlp : spacy.Language, optional
        Pre-loaded model.  Pass this to avoid reloading between calls.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``lemmas`` column.
    """
    if script not in _SCRIPT_MODEL_DEFAULTS:
        raise ValueError(
            f"Unknown script '{script}'. Choose from: "
            + ", ".join(_SCRIPT_MODEL_DEFAULTS)
        )

    if nlp is None:
        import spacy
        nlp = spacy.load(_SCRIPT_MODEL_DEFAULTS[script])

    lang_label = script.capitalize()
    print(f"Starting {lang_label} lemmatization with spaCy...")
    df["lemmas"] = df[text_col].apply(lambda t: _lemmatize(t, nlp))
    print(f"Completed! Processed {len(df)} {lang_label} posts")
    return df
