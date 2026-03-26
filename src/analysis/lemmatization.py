"""Lemmatization helpers for Russian, Greek, and English corpora."""

from __future__ import annotations

import ast
import logging

import stanza
import spacy
import pandas as pd

from src.config import (
    CYRILLIC_LEMMATIZED_CSV,
    GREEK_LEMMATIZED_CSV,
    LATIN_LEMMATIZED_CSV,
)

logger = logging.getLogger(__name__)


# Pipeline cache.
_NLP_RU: stanza.Pipeline | None = None
_NLP_EL: stanza.Pipeline | None = None
_NLP_EN: spacy.Language | None = None


def _get_ru_pipeline() -> stanza.Pipeline:
    global _NLP_RU
    if _NLP_RU is None:
        _NLP_RU = stanza.Pipeline("ru", processors="tokenize,pos,lemma",
                           tokenize_batch_size=64, lemma_batch_size=64,
                           use_gpu=False, verbose=False)
    return _NLP_RU


def _get_el_pipeline() -> stanza.Pipeline:
    global _NLP_EL
    if _NLP_EL is None:
        _NLP_EL = stanza.Pipeline("el", processors="tokenize,pos,lemma",
                           tokenize_batch_size=64, lemma_batch_size=64,
                           use_gpu=False, verbose=False)
    return _NLP_EL


def _get_en_pipeline() -> spacy.Language:
    global _NLP_EN
    if _NLP_EN is None:
        _NLP_EN = spacy.load("en_core_web_lg")
    return _NLP_EN


# Stop word lists.
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


# Shared lemma extractor.

def _extract_lemmas(doc, stopwords: frozenset[str]) -> list[str]:
    """Extract filtered lemmas from a stanza Document."""
    return [
        word.lemma.lower()
        for sent in doc.sentences
        for word in sent.words
        if word.lemma
        and word.lemma.lower() not in stopwords
        and word.upos not in ("PUNCT", "SYM", "X")
        and word.text.isalpha()
        and len(word.text) > 2
    ]


# Single-text lemmatization stubs.

def lemmatize_russian(text: str | float) -> list[str]:
    """Lemmatize one Russian string via stanza ru pipeline."""
    raise NotImplementedError(
        "Do not call lemmatize_russian() directly — it processes one doc at a time. "
        "Use lemmatize_column(df) for DataFrame input."
    )


def lemmatize_greek(text: str | float) -> list[str]:
    """Lemmatize one Modern Greek string via stanza el pipeline."""
    raise NotImplementedError(
        "Do not call lemmatize_greek() directly — it processes one doc at a time. "
        "Use lemmatize_greek_column(df) for DataFrame input."
    )


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


# DataFrame-level helpers.

def _load_existing_lemmas(csv_path) -> tuple[frozenset[str], pd.DataFrame]:
    """Load cached lemmatized rows and known message IDs."""
    from pathlib import Path
    p = Path(csv_path)
    if not p.exists():
        return frozenset(), pd.DataFrame()
    existing = pd.read_csv(p)
    if existing.empty or "message_id" not in existing.columns:
        return frozenset(), pd.DataFrame()
    # Restore list values if serialized as strings.
    if "lemmas" in existing.columns:
        existing["lemmas"] = existing["lemmas"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else v
        )
    return frozenset(existing["message_id"].astype(str)), existing

def _bulk_lemmatize(
    df: pd.DataFrame,
    text_col: str,
    pipeline: stanza.Pipeline,
    stopwords: frozenset[str],
) -> list[list[str]]:
    """Run stanza bulk_process over all rows and extract lemmas."""
    texts = [
        str(t) if not pd.isna(t) and str(t).strip() != "" else ""
        for t in df[text_col]
    ]
    # bulk_process skips empty strings, so track indices.
    non_empty_indices = [i for i, t in enumerate(texts) if t]
    non_empty_texts = [texts[i] for i in non_empty_indices]

    results: list[list[str]] = [[] for _ in range(len(texts))]
    if non_empty_texts:
        docs = pipeline.bulk_process(non_empty_texts)
        for idx, doc in zip(non_empty_indices, docs):
            results[idx] = _extract_lemmas(doc, stopwords)
    return results


def lemmatize_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
) -> pd.DataFrame:
    """Add a Russian lemmas column, reusing cached rows when available."""
    existing_ids, existing_df = _load_existing_lemmas(CYRILLIC_LEMMATIZED_CSV)
    new_df = df[~df["message_id"].astype(str).isin(existing_ids)].copy()
    logger.info(
        "Russian lemmatization (stanza bulk_process) — "
        "%d new rows (%d already lemmatized, skipping)",
        len(new_df), len(existing_ids),
    )
    if not new_df.empty:
        new_df["lemmas"] = _bulk_lemmatize(new_df, text_col, _get_ru_pipeline(), _RU_STOPWORDS)
    result = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    logger.info("Done. %d total rows.", len(result))
    return result


def lemmatize_greek_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
) -> pd.DataFrame:
    """Add a Greek lemmas column, reusing cached rows when available."""
    existing_ids, existing_df = _load_existing_lemmas(GREEK_LEMMATIZED_CSV)
    new_df = df[~df["message_id"].astype(str).isin(existing_ids)].copy()
    logger.info(
        "Greek lemmatization (stanza bulk_process) — "
        "%d new rows (%d already lemmatized, skipping)",
        len(new_df), len(existing_ids),
    )
    if not new_df.empty:
        new_df["lemmas"] = _bulk_lemmatize(new_df, text_col, _get_el_pipeline(), _EL_STOPWORDS)
    result = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    logger.info("Done. %d total rows.", len(result))
    return result


def lemmatize_english_column(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
) -> pd.DataFrame:
    """Add an English lemmas column, reusing cached rows when available."""
    existing_ids, existing_df = _load_existing_lemmas(LATIN_LEMMATIZED_CSV)
    new_df = df[~df["message_id"].astype(str).isin(existing_ids)].copy()
    logger.info(
        "English lemmatization (spaCy pipe) — "
        "%d new rows (%d already lemmatized, skipping)",
        len(new_df), len(existing_ids),
    )
    if not new_df.empty:
        nlp = _get_en_pipeline()
        texts = new_df[text_col].fillna("").tolist()
        new_df["lemmas"] = [
            [t.lemma_.lower() for t in doc
             if not t.is_stop and not t.is_punct
             and not t.is_space and t.is_alpha and len(t.text) > 2]
            for doc in nlp.pipe(texts, batch_size=128)
        ]
    result = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    logger.info("Done. %d total rows.", len(result))
    return result
