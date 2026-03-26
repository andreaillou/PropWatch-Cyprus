"""Text cleaning and script-type classification utilities."""

from __future__ import annotations

import logging
import re

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)


# Build detector once at import time.
_DETECTOR = (
    LanguageDetectorBuilder.from_languages(
        Language.RUSSIAN, Language.GREEK, Language.ENGLISH
    ).build()
)


def clean_text(text) -> str:
    """Normalize a single message string."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\*\*|__|~~|`", "", text)
    text = re.sub(r"[^\w\s\.,!?:\"'\-\u2014\u00ab\u00bb]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    """Return dominant language using lingua-py."""
    if pd.isna(text) or str(text).strip() == "":
        return "unknown"
    result = _DETECTOR.detect_language_of(str(text))
    if result is None:
        return "unknown"
    return result.name.lower()  # e.g. Language.RUSSIAN -> "russian"


def classify_script(text: str) -> str:
    """Classify *text* as ``'cyrillic'``, ``'latin'``, ``'greek'``, or ``'unknown'``."""
    if pd.isna(text) or text == "":
        return "unknown"
    cyrillic = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    greek = len(re.findall(r"[α-ωΑ-Ωά-ώΆ-Ώ]", text))
    counts = {"cyrillic": cyrillic, "latin": latin, "greek": greek}
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    return dominant if counts[dominant] > 0 else "unknown"


def clean_and_split(
    df: pd.DataFrame,
    text_col: str = "text",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean text and split into Russian, English, and Greek dataframes."""
    df["text_cleaned"] = df[text_col].apply(clean_text)
    df["script_type"]  = df["text_cleaned"].apply(classify_script)
    df["language"]     = df["text_cleaned"].apply(detect_language)

    russian_df = df[df["language"] == "russian"].copy()
    english_df = df[df["language"] == "english"].copy()
    greek_df = df[df["language"] == "greek"].copy()

    logger.info(
        "Language split: %d Russian | %d English | %d Greek",
        len(russian_df), len(english_df), len(greek_df),
    )
    return df, russian_df, english_df, greek_df
