"""Lemmatization for all three sub-corpora.

Language assignment:
- Russian  → stanza ``ru`` pipeline  (handles morphological case inflection)
- Greek    → stanza ``el`` pipeline  (outperforms spaCy on Modern Greek)
- English  → spaCy ``en_core_web_lg`` (Track A features; RT/Sputnik English content)

Do NOT use spaCy for Russian or Greek. Do NOT use stanza for English.

First-time setup (run once):
    import stanza; stanza.download("ru"); stanza.download("el")
    python -m spacy download en_core_web_lg
"""

from __future__ import annotations

import stanza
import spacy
import pandas as pd


# ── Pipeline cache — load once, reuse everywhere ──────────────────────────────
_NLP_RU: stanza.Pipeline | None = None
_NLP_EL: stanza.Pipeline | None = None
_NLP_EN: spacy.Language | None = None


def _get_ru_pipeline() -> stanza.Pipeline:
    global _NLP_RU
    if _NLP_RU is None:
        _NLP_RU = stanza.Pipeline("ru", processors="tokenize,pos,lemma",
                                  use_gpu=False, verbose=False)
    return _NLP_RU


def _get_el_pipeline() -> stanza.Pipeline:
    global _NLP_EL
    if _NLP_EL is None:
        _NLP_EL = stanza.Pipeline("el", processors="tokenize,pos,lemma",
                                  use_gpu=False, verbose=False)
    return _NLP_EL


def _get_en_pipeline() -> spacy.Language:
    global _NLP_EN
    if _NLP_EN is None:
        _NLP_EN = spacy.load("en_core_web_lg")
    return _NLP_EN


# ── Stop word lists ───────────────────────────────────────────────────────────
_RU_STOPWORDS: frozenset[str] = frozenset({
    "и", "в", "не", "на", "что", "с", "по", "как", "это", "к",
    "из", "но", "за", "то", "до", "же", "от", "а", "или", "об",
    "для", "при", "так", "быть", "он", "она", "они", "мы", "вы",
    "его", "её", "их", "наш", "ваш", "этот", "тот", "все",
    "уже", "ещё", "даже", "только", "если", "когда", "чтобы",
})

_EL_STOPWORDS: frozenset[str] = frozenset({
    "και", "το", "τα", "τη", "τον", "την", "τους", "τις", "της",
    "με", "για", "από", "στο", "στη", "στον", "στην", "στα",
    "που", "να", "θα", "αλλά", "ή", "αν", "ως", "ότι", "είναι",
    "δεν", "μας", "σας", "τους", "αυτό", "αυτή", "αυτός", "κι",
})


# ── Single-text lemmatization ─────────────────────────────────────────────────

def lemmatize_russian(text: str | float) -> list[str]:
    """Lemmatize one Russian string via stanza ru pipeline."""
    if pd.isna(text) or str(text).strip() == "":
        return []
    doc = _get_ru_pipeline()(str(text))
    return [
        word.lemma.lower()
        for sent in doc.sentences
        for word in sent.words
        if word.lemma
        and word.lemma.lower() not in _RU_STOPWORDS
        and word.upos not in ("PUNCT", "SYM", "X")
        and word.text.isalpha()
        and len(word.text) > 2
    ]


def lemmatize_greek(text: str | float) -> list[str]:
    """Lemmatize one Modern Greek string via stanza el pipeline."""
    if pd.isna(text) or str(text).strip() == "":
        return []
    doc = _get_el_pipeline()(str(text))
    return [
        word.lemma.lower()
        for sent in doc.sentences
        for word in sent.words
        if word.lemma
        and word.lemma.lower() not in _EL_STOPWORDS
        and word.upos not in ("PUNCT", "SYM", "X")
        and word.text.isalpha()
        and len(word.text) > 2
    ]


def lemmatize_english(text: str | float) -> list[str]:
    """Lemmatize one English string via spaCy en_core_web_lg."""
    if pd.isna(text) or str(text).strip() == "":
        return []
    doc = _get_en_pipeline()(str(text))
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
        and len(token.text) > 2
    ]


# ── DataFrame-level helpers ───────────────────────────────────────────────────

def lemmatize_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
) -> pd.DataFrame:
    """Add ``lemmas`` column to *df* using Russian stanza lemmatization."""
    print(f"Russian lemmatization (stanza) — {len(df)} rows...")
    df["lemmas"] = df[text_col].apply(lemmatize_russian)
    print(f"Done. {len(df)} rows lemmatized.")
    return df


def lemmatize_greek_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
) -> pd.DataFrame:
    """Add ``lemmas`` column to *df* using Greek stanza lemmatization."""
    print(f"Greek lemmatization (stanza) — {len(df)} rows...")
    df["lemmas"] = df[text_col].apply(lemmatize_greek)
    print(f"Done. {len(df)} rows lemmatized.")
    return df


def lemmatize_english_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
) -> pd.DataFrame:
    """Add ``lemmas`` column to *df* using English spaCy lemmatization.

    Used for RT/Sputnik English articles (Tier 1) and English Telegram posts.
    Requires ``en_core_web_lg``:
        python -m spacy download en_core_web_lg
    """
    print(f"English lemmatization (spaCy) — {len(df)} rows...")
    df["lemmas"] = df[text_col].apply(lemmatize_english)
    print(f"Done. {len(df)} rows lemmatized.")
    return df
