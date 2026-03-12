"""Filtering pipeline for scraped posts.

Applies a multi-stage pipeline:
1. Remove short messages (< 20 characters).
2. Remove navigation / UI dumps.
3. Remove spam / ads using an exclusion keyword list.
4. Zero-shot NLI classification — keep only posts the model considers
   politically / socially / culturally relevant.
5. De-duplicate by text.
6. Tag surviving posts with keyword categories (no removal — posts with
   zero keyword hits stay with 0 across all category columns).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from transformers import Pipeline  # only for type hints

logger = logging.getLogger(__name__)

# ── Exclusion keywords (anti-spam) ────────────────────────────────────────────
# Kept empty for now; uncomment lines below as needed.
EXCLUDE_KEYWORDS: list[str] = [
     "usdt", "btc", "crypto", "binance", "exchange",
     "дайджест", "digest"
    # "продам", "куплю", "сдам", "аренда", "квартир", "вилл",
    # "обмен", "наличны", "кэш", "курс", "доставк", "отправк",
    # "цена", "стоимость", "price", "cost", "deal",
    # "whatsapp", "pm", "dm", "личк", "директ", "лс",
]

# ── Inclusion keywords (political topics) ────────────────────────────────────
INCLUDE_KEYWORDS: dict[str, list[str]] = {
    "WAR": [
        # English / Latin
        r"war", r"invasion", r"conflict", r"military", r"army",
        r"frontline", r"attack",
        # Russian / Cyrillic
        r"войн", r"воен", r"арми", r"фронт", r"наступлен", r"атак",
        r"\bсво\b", r"спецоперац",
        # Greek
        r"πόλεμ", r"εισβολ", r"σύγκρουσ", r"στρατ", r"μέτωπ", r"επίθεσ",
    ],
    "KEY_ACT": [
        # English / Latin
        r"putin", r"zelensky", r"biden", r"nato", r"eu", r"europe",
        r"kremlin", r"moscow", r"kiev", r"kyiv", r"trump",
        # Russian / Cyrillic
        r"путин", r"зеленск", r"байден", r"нато", r"\bес\b", r"европ", r"ястреб",
        r"кремл", r"москв", r"киев", r"росси", r"укоаин", r"\bрф\b",
        r"трамп",
        # Greek
        r"πούτιν", r"ζελένσκ", r"μπάιντεν", r"νατο", r"\bεε\b", r"ευρώπ",
        r"κρεμλίν", r"μόσχ", r"κίεβ", r"ρωσί", r"ουκραιν", r"τραμπ",
    ],
    "IDEAL_TER": [
        # English / Latin
        r"nazi", r"fascist", r"propaganda", r"fake", r"truth", r"west",
        r"imperial", r"russophobia",
        # Russian / Cyrillic
        r"наци", r"фашист", r"пропаганд", r"фейк", r"правд", r"запад",
        r"импер", r"русофоб", r"освобожд",
        # Greek
        r"ναζ", r"φασίστ", r"προπαγάνδ", r"ψεύδ", r"αλήθει",
        r"δύσ", r"ιμπεριαλ", r"ρωσοφοβ",
    ],
    # ── Cleavage codes (Layer 2 annotation schema) ────────────────────────
    "CY_DIV": [
        # English / Latin
        r"cyprus problem", r"reunification", r"occupied", r"buffer zone",
        # Greek
        r"κυπριακό", r"επανένωσ", r"κατεχόμεν", r"πράσινη γραμμή",
        # Russian / Cyrillic
        r"кипрск", r"разделени", r"оккупир", r"турецк",
    ],
    "EU_SKEP": [
        # English / Latin
        r"brussels", r"eu sanctions", r"sovereignty",
        # Greek
        r"βρυξέλλ", r"κυριαρχί", r"κυρώσεις", r"ευρωπαϊκ",
        # Russian / Cyrillic
        r"брюссел", r"суверенит", r"санкц", r"евросоюз",
    ],
    "BAIL_IN": [
        # English / Latin
        r"bail.in", r"haircut", r"imf", r"bank levy",
        # Greek
        r"κούρεμα", r"κυπριακή τράπεζα", r"ΔΝΤ",
        # Russian / Cyrillic
        r"стрижк", r"кипрск банк", r"мвф",
    ],
    "ORTHO": [
        # English / Latin
        r"orthodox", r"church", r"civilisation", r"christian values",
        # Greek
        r"ορθόδοξ", r"εκκλησί", r"χριστιανικ", r"πολιτισμ",
        # Russian / Cyrillic
        r"православ", r"церков", r"цивилизац", r"христианск",
    ],
    "ELIT": [
        # English / Latin
        r"corrupt elite", r"deep state", r"establishment", r"oligarch",
        # Greek
        r"ελίτ", r"διαφθορ", r"κατεστημέν", r"παρακράτ",
        # Russian / Cyrillic
        r"элит", r"коррупц", r"истеблишмент", r"олигарх",
    ],
    "MIGR": [
        # English / Latin
        r"migrant", r"refugee", r"illegal immigration", r"border",
        # Greek
        r"μετανάστ", r"πρόσφυγ", r"παράνομ", r"σύνορ",
        # Russian / Cyrillic
        r"мигрант", r"беженц", r"нелегальн", r"границ",
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


def is_navigation_dump(text) -> bool:
    """Return ``True`` if *text* looks like a scraped navigation/UI page.

    Detects two patterns that trafilatura sometimes returns instead of an
    article:

    1. **Icon-list dump** — multiple lines starting with ``"icon "``.
       Characteristic of RT / Sputnik CSS icon-font listings.
    2. **Menu/nav dump** — the majority of non-empty lines are very short
       (≤ 5 words), which indicates a navigation menu, tag page, or site
       index rather than continuous prose.
    """
    if not isinstance(text, str):
        return False

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    # Heuristic 1: icon-font dump (≥4 lines starting with "icon ")
    icon_lines = sum(1 for ln in lines if ln.lower().startswith("icon "))
    if icon_lines >= 4:
        return True

    # Heuristic 2: nav/menu dump — >55 % of lines have ≤5 words AND
    # there are at least 15 lines (short texts are handled by length filter)
    if len(lines) >= 15:
        short = sum(1 for ln in lines if len(ln.split()) <= 5)
        if short / len(lines) > 0.55:
            return True

    return False


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


# ── Zero-shot NLI political-relevance gate ────────────────────────────────────

# Tone-neutral hypothesis used for zero-shot NLI classification.
# The wording avoids value-laden terms; it simply asks whether the text
# relates to topics a political scientist would study.
NLI_HYPOTHESIS: str = (
    "This text discusses a political, geopolitical, social, cultural, "
    "or public-policy issue."
)

# Model identifier — multilingual NLI, covers EN / EL / RU.
NLI_MODEL_NAME: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Entailment probability threshold.  Posts scoring below this are dropped.
NLI_THRESHOLD: float = 0.5

# Maximum token length fed to the NLI model (truncated, not split).
_NLI_MAX_LENGTH: int = 512

# Batch size for NLI inference (tune for your GPU / RAM).
_NLI_BATCH_SIZE: int = 32

# Singleton cache so the model is loaded only once per session.
_nli_pipeline: Pipeline | None = None


def _get_nli_pipeline() -> Pipeline:
    """Lazy-load and cache the zero-shot-classification pipeline."""
    global _nli_pipeline  # noqa: PLW0603
    if _nli_pipeline is None:
        from transformers import pipeline as hf_pipeline

        _nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL_NAME,
            device="cpu",          # change to 0 / "cuda" if GPU available
        )
    return _nli_pipeline


def nli_score(text: str) -> float:
    """Return the entailment probability that *text* is politically relevant."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    clf = _get_nli_pipeline()
    # Truncate to first ~_NLI_MAX_LENGTH chars (≈tokens) to stay within
    # the model's context window without expensive tokeniser overhead.
    result = clf(
        text[:_NLI_MAX_LENGTH * 4],
        candidate_labels=["political or social topic", "other"],
        hypothesis_template="This text is about {}.",
    )
    # result["labels"] is ordered by score; find the political label's score.
    for label, score in zip(result["labels"], result["scores"]):
        if label == "political or social topic":
            return float(score)
    return 0.0


def nli_scores_batch(
    texts: list[str],
    batch_size: int = _NLI_BATCH_SIZE,
) -> np.ndarray:
    """Score a list of texts in batches.  Returns an array of floats."""
    clf = _get_nli_pipeline()
    scores: list[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="NLI scoring"):
        batch = [
            (t[:_NLI_MAX_LENGTH * 4] if isinstance(t, str) and t.strip() else "")
            for t in texts[i : i + batch_size]
        ]
        results = clf(
            batch,
            candidate_labels=["political or social topic", "other"],
            hypothesis_template="This text is about {}.",
        )
        # When batch size == 1, transformers may return a dict instead of list.
        if isinstance(results, dict):
            results = [results]
        for r in results:
            for label, score in zip(r["labels"], r["scores"]):
                if label == "political or social topic":
                    scores.append(float(score))
                    break
            else:
                scores.append(0.0)
    return np.array(scores)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def filter_messages(
    df: pd.DataFrame,
    text_col: str = "text",
    nli_threshold: float = NLI_THRESHOLD,
    nli_batch_size: int = _NLI_BATCH_SIZE,
) -> pd.DataFrame:
    """Run the filtering pipeline.

    Steps
    -----
    1. Length filter (< 20 chars).
    2. Navigation / UI dump filter.
    3. Exclusion-keyword spam filter.
    4. Zero-shot NLI political-relevance gate.
    5. De-duplication.

    After filtering, ``tag_categories`` should be called on the result to
    add keyword-based category columns (no posts are removed at that stage).
    """
    n0 = len(df)

    # Step 1 – Length filter
    df = df[df[text_col].str.len() > 20].copy()
    logger.info(
        "Step 1  (Length filter):      %7d rows  (−%d)",
        len(df), n0 - len(df),
    )

    # Step 2 – Remove navigation / UI dumps
    n_prev = len(df)
    df = df[~df[text_col].apply(is_navigation_dump)].copy()
    logger.info(
        "Step 2  (Nav-dump filter):    %7d rows  (−%d)",
        len(df), n_prev - len(df),
    )

    # Step 3 – Remove exclusion-keyword spam
    n_prev = len(df)
    df = df[~df[text_col].apply(has_exclusion)].copy()
    logger.info(
        "Step 3  (Spam filter):        %7d rows  (−%d)",
        len(df), n_prev - len(df),
    )

    # Step 4 – Zero-shot NLI political-relevance gate
    n_prev = len(df)
    texts = df[text_col].tolist()
    scores = nli_scores_batch(texts, batch_size=nli_batch_size)
    df["nli_score"] = scores
    df = df[df["nli_score"] >= nli_threshold].copy()
    logger.info(
        "Step 4  (NLI gate >=%.2f):    %7d rows  (−%d)",
        nli_threshold, len(df), n_prev - len(df),
    )

    # Step 5 – De-duplicate
    n_prev = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first").copy()
    logger.info(
        "Step 5  (Deduplicate):        %7d rows  (−%d)",
        len(df), n_prev - len(df),
    )

    return df


def tag_categories(
    df: pd.DataFrame,
    text_col: str = "text",
    categories: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Add binary indicator columns for each keyword category.

    Every post that passed ``filter_messages`` is kept regardless of whether
    it matches any keyword category.  Posts with no keyword hits simply
    receive ``0`` across all category columns.
    """
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
