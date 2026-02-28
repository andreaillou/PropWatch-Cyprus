"""Text cleaning and script-type classification utilities."""

import re

import pandas as pd
from lingua import Language, LanguageDetectorBuilder


# ── Lingua-py language detector (built once at import time) ─────────────────────
# Building the detector is expensive; keep it as a module-level singleton.
_DETECTOR = (
    LanguageDetectorBuilder.from_languages(
        Language.RUSSIAN, Language.GREEK, Language.ENGLISH
    ).build()
)


def clean_text(text) -> str:
    """Normalize a single message string.

    * Strips URLs and Telegram markdown.
    * Removes emojis / non-standard characters (keeps basic punctuation).
    * Collapses whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\*\*|__|~~|`", "", text)
    text = re.sub(r"[^\w\s\.,!?:\"'\-\u2014\u00ab\u00bb]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    """Return the dominant language of *text* using lingua-py.

    Returns
    -------
    str
        ``'russian'``, ``'greek'``, ``'english'``, or ``'unknown'`` when
        lingua-py cannot confidently identify the language (e.g. very short
        strings or code-switched text).

    Notes
    -----
    This is the **authoritative** language label for downstream NLP models.
    ``classify_script`` is kept as a fast secondary signal only.
    """
    if pd.isna(text) or str(text).strip() == "":
        return "unknown"
    result = _DETECTOR.detect_language_of(str(text))
    if result is None:
        return "unknown"
    return result.name.lower()  # e.g. Language.RUSSIAN -> "russian"



    """Classify *text* as ``'cyrillic'``, ``'latin'``, ``'greek'``, or ``'unknown'``."""
    if pd.isna(text) or text == "":
        return "unknown"
    cyrillic = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin    = len(re.findall(r"[a-zA-Z]", text))
    greek    = len(re.findall(r"[α-ωΑ-Ωά-ώΆ-Ώ]", text))
    counts   = {"cyrillic": cyrillic, "latin": latin, "greek": greek}
    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 0 else "unknown"


def clean_and_split(
    df: pd.DataFrame,
    text_col: str = "text",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean the text column and split by detected language.

    Adds two classification columns:

    * ``script_type`` – fast character-count heuristic
      (``'cyrillic'`` / ``'greek'`` / ``'latin'`` / ``'unknown'``).
    * ``language``    – authoritative lingua-py detection
      (``'russian'`` / ``'greek'`` / ``'english'`` / ``'unknown'``).

    Splits are based on ``language`` (the authoritative signal).

    Returns ``(full_df, russian_df, english_df, greek_df)``.
    """
    df["text_cleaned"] = df[text_col].apply(clean_text)
    df["script_type"]  = df["text_cleaned"].apply(classify_script)
    df["language"]     = df["text_cleaned"].apply(detect_language)

    russian_df = df[df["language"] == "russian"].copy()
    english_df = df[df["language"] == "english"].copy()
    greek_df   = df[df["language"] == "greek"].copy()

    print(f"Russian posts: {len(russian_df)}")
    print(f"English posts: {len(english_df)}")
    print(f"Greek posts:   {len(greek_df)}")
    return df, russian_df, english_df, greek_df
