"""Text cleaning and script-type classification utilities."""

import re

import pandas as pd


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


def classify_script(text) -> str:
    """Classify *text* as ``'cyrillic'``, ``'latin'``, or ``'unknown'``."""
    if pd.isna(text) or text == "":
        return "unknown"
    cyrillic = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    if cyrillic == 0 and latin == 0:
        return "unknown"
    return "cyrillic" if cyrillic > latin else "latin"


def clean_and_split(
    df: pd.DataFrame,
    text_col: str = "text",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean the text column and split by script type.

    Returns ``(full_df, cyrillic_df, latin_df)``.
    """
    df["text_cleaned"] = df[text_col].apply(clean_text)
    df["script_type"] = df["text_cleaned"].apply(classify_script)

    cyrillic_df = df[df["script_type"] == "cyrillic"].copy()
    latin_df = df[df["script_type"] == "latin"].copy()

    print(f"Cyrillic posts: {len(cyrillic_df)}")
    print(f"Latin posts:    {len(latin_df)}")
    return df, cyrillic_df, latin_df
