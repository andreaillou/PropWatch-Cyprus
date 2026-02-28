"""Keyword-based filtering for Telegram posts.

Applies a three-stage pipeline:
1. Remove short messages (< 20 characters).
2. Remove spam / ads using an exclusion keyword list.
3. Keep only politically relevant messages using an inclusion keyword list.
4. De-duplicate by text.
"""

import re

import pandas as pd

# ── Exclusion keywords (anti-spam) ────────────────────────────────────────────
# Kept empty for now; uncomment lines below as needed.
EXCLUDE_KEYWORDS: list[str] = [
    # "usdt", "btc", "crypto", "binance", "exchange",
    # "rent", "sale", "apartment", "villa", "flat", "bedroom",
    # "продам", "куплю", "сдам", "аренда", "квартир", "вилл",
    # "обмен", "наличны", "кэш", "курс", "доставк", "отправк",
    # "цена", "стоимость", "price", "cost", "deal",
    # "whatsapp", "pm", "dm", "личк", "директ", "лс",
]

# ── Inclusion keywords (political topics) ────────────────────────────────────
INCLUDE_KEYWORDS: dict[str, list[str]] = {
    "War_Conflict": [
        r"war", r"invasion", r"conflict", r"military", r"army",
        r"frontline", r"attack",
        r"войн", r"воен", r"арми", r"фронт", r"наступлен", r"атак",
        r"\bсво\b", r"спецоперац",
    ],
    "Key_Actors": [
        r"putin", r"zelensky", r"biden", r"nato", r"eu", r"europe",
        r"kremlin", r"moscow", r"kiev", r"kyiv", r"trump",
        r"путин", r"зеленск", r"байден", r"нато", r"\bес\b", r"европ",
        r"кремл", r"москв", r"киев", r"росси", r"укоаин", r"\bрф\b",
        r"трамп",
    ],
    "Ideological_Terms": [
        r"nazi", r"fascist", r"propaganda", r"fake", r"truth", r"west",
        r"imperial", r"russophobia",
        r"наци", r"фашист", r"пропаганд", r"фейк", r"правд", r"запад",
        r"импер", r"русофоб", r"освобожд",
    ],
}


# ── Helper predicates ─────────────────────────────────────────────────────────

def _matches_any(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    for kw in keywords:
        if r"\b" in kw:
            if re.search(kw, text_lower):
                return True
        else:
            if kw in text_lower:
                return True
    return False


def has_exclusion(text) -> bool:
    """Return ``True`` if *text* contains any exclusion keyword."""
    if not isinstance(text, str):
        return False
    return _matches_any(text, EXCLUDE_KEYWORDS)


def has_inclusion(text) -> bool:
    """Return ``True`` if *text* matches any inclusion keyword category."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    for patterns in INCLUDE_KEYWORDS.values():
        for p in patterns:
            if r"\b" in p:
                if re.search(p, text_lower):
                    return True
            else:
                if p in text_lower:
                    return True
    return False


# ── Main pipeline ─────────────────────────────────────────────────────────────

def filter_messages(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Run the full four-step filtering pipeline.

    Returns a new DataFrame containing only unique, politically relevant
    messages.
    """
    # Step 1 – Length filter
    df = df[df[text_col].str.len() > 20].copy()
    print(f"Step 1 (Length Filter): {len(df)} rows remain")

    # Step 2 – Remove ads
    df = df[~df[text_col].apply(has_exclusion)].copy()
    print(f"Step 2 (Remove Ads):    {len(df)} rows remain")

    # Step 3 – Keep political topics
    df = df[df[text_col].apply(has_inclusion)].copy()
    print(f"Step 3 (Topic Filter):  {len(df)} rows remain")

    # Step 4 – De-duplicate
    df = df.drop_duplicates(subset=[text_col], keep="first").copy()
    print(f"Step 4 (Deduplicate):   {len(df)} unique rows remain")

    return df


def tag_categories(
    df: pd.DataFrame,
    text_col: str = "text",
    categories: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Add binary indicator columns for each keyword category."""
    if categories is None:
        categories = INCLUDE_KEYWORDS

    for category, patterns in categories.items():
        df[category] = df[text_col].apply(
            lambda x, pats=patterns: (
                1
                if isinstance(x, str)
                and any(
                    (re.search(p, x.lower()) if r"\b" in p else p in x.lower())
                    for p in pats
                )
                else 0
            )
        )
    return df
