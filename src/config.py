"""Centralized project configuration.

All file paths are relative to the repository root so the project works on any
machine.  Environment variables (Telegram credentials) are loaded from a `.env`
file via ``python-dotenv``.
"""

import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

# ── Repository root ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Telegram API ─────────────────────────────────────────────────────────────
TELEGRAM_APP_ID = int(os.getenv("TELEGRAM_APP_ID", "0"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")

# ── Scraping parameters ─────────────────────────────────────────────────────
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
MESSAGE_LIMIT = 3000

CHANNELS = {
    "Cyprus": [
        "rusembcy",
    ],
}

# ── Data directories ─────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "cyprus_data"
TELEGRAM_DIR = DATA_DIR / "telegram" / "russian_embassy"

# Ensure directories exist at import time
TELEGRAM_DIR.mkdir(parents=True, exist_ok=True)

# ── Output file paths ────────────────────────────────────────────────────────
RAW_CSV = TELEGRAM_DIR / "telegram_posts_cyprus_raw.csv"
PRECLEANED_CSV = TELEGRAM_DIR / "telegram_posts_cyprus_precleaned.csv"
CLEAN_CSV = TELEGRAM_DIR / "telegram_posts_cyprus_clean.csv"

# ── Per-script split CSVs ─────────────────────────────────────────────────────
CYRILLIC_CSV = TELEGRAM_DIR / "cyrillic_posts.csv"
LATIN_CSV    = TELEGRAM_DIR / "latin_posts.csv"
GREEK_CSV    = TELEGRAM_DIR / "greek_posts.csv"

# ── Lemmatized CSVs ───────────────────────────────────────────────────────────
CYRILLIC_LEMMATIZED_CSV = TELEGRAM_DIR / "cyrillic_posts_lemmatized.csv"
LATIN_LEMMATIZED_CSV    = TELEGRAM_DIR / "latin_posts_lemmatized.csv"
GREEK_LEMMATIZED_CSV    = TELEGRAM_DIR / "greek_posts_lemmatized.csv"

# ── Frequency / n-gram CSVs ───────────────────────────────────────────────────
WORD_FREQ_CSV   = TELEGRAM_DIR / "russian_word_frequency.csv"
BIGRAMS_CSV     = TELEGRAM_DIR / "russian_bigrams.csv"
TRIGRAMS_CSV    = TELEGRAM_DIR / "russian_trigrams.csv"

LATIN_WORD_FREQ_CSV = TELEGRAM_DIR / "latin_word_frequency.csv"
LATIN_BIGRAMS_CSV   = TELEGRAM_DIR / "latin_bigrams.csv"
LATIN_TRIGRAMS_CSV  = TELEGRAM_DIR / "latin_trigrams.csv"

GREEK_WORD_FREQ_CSV = TELEGRAM_DIR / "greek_word_frequency.csv"
GREEK_BIGRAMS_CSV   = TELEGRAM_DIR / "greek_bigrams.csv"
GREEK_TRIGRAMS_CSV  = TELEGRAM_DIR / "greek_trigrams.csv"

# ── Model directory (placeholder) ────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
