"""Lemmatization for Russian and Greek sub-corpora using stanza.

stanza is used for both languages because it handles morphological case
inflection better than spaCy on Russian (ru pipeline) and outperforms
spaCy on Modern Greek benchmarks (el pipeline).

NOTE: spaCy is intentionally NOT used here. It remains in the project
only for English text and Track A syntactic feature extraction
(passive voice ratio, imperative verb ratio). Do not replace or merge
those concerns into this module.

First-time setup (run once in your environment):
    import stanza
    stanza.download("ru")
    stanza.download("el")

TODO (future quality upgrade): For Greek lemmatization, consider
replacing stanza with GR-NLP-TOOLKIT (Greek-BERT-based), which
outperforms stanza on Modern Greek morphological tagging. Hold off
until the pipeline is stable — it is a niche library.
"""

from __future__ import annotations

import stanza
import pandas as pd


# ── Module-level pipeline cache ───────────────────────────────────────────────
# Pipelines are expensive to initialise; load once and reuse.
_NLP_RU: stanza.Pipeline | None = None
_NLP_EL: stanza.Pipeline | None = None


def _get_ru_pipeline() -> stanza.Pipeline:
    global _NLP_RU
    if _NLP_RU is None:
        _NLP_RU = stanza.Pipeline(
            "ru",
            processors="tokenize,pos,lemma",
            use_gpu=False,   # set True if running on GPU
            verbose=False,
        )
    return _NLP_RU


def _get_el_pipeline() -> stanza.Pipeline:
    global _NLP_EL
    if _NLP_EL is None:
        _NLP_EL = stanza.Pipeline(
            "el",
            processors="tokenize,pos,lemma",
            use_gpu=False,
            verbose=False,
        )
    return _NLP_EL


# ── Russian stop words (minimal functional list) ──────────────────────────────
# stanza does not expose a stop word list directly.
# This covers the most common Russian function words.
_RU_STOPWORDS: frozenset[str] = frozenset({
    "и", "в", "не", "на", "что", "с", "по", "как", "это", "к",
    "из", "но", "за", "то", "до", "же", "от", "а", "или", "об",
    "для", "при", "так", "быть", "он", "она", "они", "мы", "вы",
    "его", "её", "их", "наш", "ваш", "этот", "тот", "все",
    "уже", "ещё", "даже", "только", "если", "когда", "чтобы",
})

# ── Greek stop words (minimal functional list) ────────────────────────────────
_EL_STOPWORDS: frozenset[str] = frozenset({
    "και", "το", "τα", "τη", "τον", "την", "τους", "τις", "της",
    "με", "για", "από", "στο", "στη", "στον", "στην", "στα",
    "που", "να", "θα", "αλλά", "ή", "αν", "ως", "ότι", "είναι",
    "δεν", "μας", "σας", "τους", "αυτό", "αυτή", "αυτός", "κι",
})


# ── Core lemmatization functions ──────────────────────────────────────────────────

def lemmatize_russian(text: str | float, nlp: stanza.Pipeline | None = None) -> list[str]:
    """Lemmatize a single Russian text string.

    Parameters
    ----------
    text : str or NaN
        Input text (raw or cleaned).
    nlp : stanza.Pipeline, optional
        A pre-loaded stanza Russian pipeline. If None, uses the
        module-level cached pipeline (preferred).

    Returns
    -------
    list[str]
        Filtered lemmas: lowercase, alphabetic, length > 2,
        not in Russian stop word list.
    """
    if pd.isna(text) or str(text).strip() == "":
        return []

    pipeline = nlp if nlp is not None else _get_ru_pipeline()
    doc = pipeline(str(text))

    return [
        word.lemma.lower()
        for sentence in doc.sentences
        for word in sentence.words
        if word.lemma
        and word.lemma.lower() not in _RU_STOPWORDS
        and word.upos not in ("PUNCT", "SYM", "X")
        and word.text.isalpha()
        and len(word.text) > 2
    ]


def lemmatize_greek(text: str | float, nlp: stanza.Pipeline | None = None) -> list[str]:
    """Lemmatize a single Modern Greek text string.

    Parameters
    ----------
    text : str or NaN
        Input text (raw or cleaned).
    nlp : stanza.Pipeline, optional
        A pre-loaded stanza Greek pipeline. If None, uses the
        module-level cached pipeline (preferred).

    Returns
    -------
    list[str]
        Filtered lemmas: lowercase, alphabetic, length > 2,
        not in Greek stop word list.
    """
    if pd.isna(text) or str(text).strip() == "":
        return []

    pipeline = nlp if nlp is not None else _get_el_pipeline()
    doc = pipeline(str(text))

    return [
        word.lemma.lower()
        for sentence in doc.sentences
        for word in sentence.words
        if word.lemma
        and word.lemma.lower() not in _EL_STOPWORDS
        and word.upos not in ("PUNCT", "SYM", "X")
        and word.text.isalpha()
        and len(word.text) > 2
    ]


# ── DataFrame-level helpers ───────────────────────────────────────────────────────

def lemmatize_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
    nlp: stanza.Pipeline | None = None,
) -> pd.DataFrame:
    """Add a ``lemmas`` column to *df* using Russian lemmatization.

    Mutates and returns *df*. The ``lemmas`` column contains a list[str]
    per row (empty list for empty/NaN input).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *text_col*.
    text_col : str
        Column of cleaned Russian text to lemmatize.
    nlp : stanza.Pipeline, optional
        Pass a pre-loaded pipeline to avoid re-loading across calls.
        If None, uses the module-level cached pipeline.
    """
    pipeline = nlp if nlp is not None else _get_ru_pipeline()
    print(f"Starting Russian lemmatization (stanza) on {len(df)} rows...")
    df["lemmas"] = df[text_col].apply(lambda t: lemmatize_russian(t, pipeline))
    print(f"Completed. {len(df)} Russian posts lemmatized.")
    return df


def lemmatize_greek_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
    nlp: stanza.Pipeline | None = None,
) -> pd.DataFrame:
    """Add a ``lemmas`` column to *df* using Greek lemmatization.

    Identical contract to :func:`lemmatize_column` but uses the
    stanza Greek (el) pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *text_col*.
    text_col : str
        Column of cleaned Greek text to lemmatize.
    nlp : stanza.Pipeline, optional
        Pass a pre-loaded pipeline to avoid re-loading across calls.
        If None, uses the module-level cached pipeline.
    """
    pipeline = nlp if nlp is not None else _get_el_pipeline()
    print(f"Starting Greek lemmatization (stanza) on {len(df)} rows...")
    df["lemmas"] = df[text_col].apply(lambda t: lemmatize_greek(t, pipeline))
    print(f"Completed. {len(df)} Greek posts lemmatized.")
    return df


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
