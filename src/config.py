"""Centralized project configuration.

All file paths are relative to the repository root so the project works on any
machine.  Environment variables (Telegram credentials) are loaded from a `.env`
file via ``python-dotenv``.
"""

import os
from pathlib import Path
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Repository root ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Telegram API ─────────────────────────────────────────────────────────────
TELEGRAM_APP_ID = int(os.getenv("TELEGRAM_APP_ID", "0"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")

# ── Twitter/X API ────────────────────────────────────────────────────────────
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# ── Scraping parameters ─────────────────────────────────────────────────────
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
MESSAGE_LIMIT = 3000

# ── Channel list (loaded from configs/channels.yaml) ─────────────────────────
_CHANNELS_FILE = ROOT_DIR / "configs" / "channels.yaml"

def _load_active_channels() -> dict[str, list[str]]:
    """Load non-archived Telegram channels from configs/channels.yaml."""
    with open(_CHANNELS_FILE) as f:
        data = yaml.safe_load(f)
    channels: dict[str, list[str]] = {}
    for tier_key, entries in data.items():
        for entry in entries:
            if not entry.get("archived", False):
                tier_label = tier_key  # e.g. "tier3_cyprus"
                channels.setdefault(tier_label, []).append(entry["handle"])
    return channels

CHANNELS = _load_active_channels()

# ── Data directories ─────────────────────────────────────────────────────────
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TELEGRAM_RAW_DIR = RAW_DIR / "telegram"
TWITTER_RAW_DIR  = RAW_DIR / "twitter"

# Ensure directories exist at import time
for _d in [TELEGRAM_RAW_DIR, TWITTER_RAW_DIR, PROCESSED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Merged corpus output paths (all channels combined) ────────────────────
RAW_CSV         = PROCESSED_DIR / "corpus_raw.csv"
PRECLEANED_CSV  = PROCESSED_DIR / "corpus_precleaned.csv"
CLEAN_CSV       = PROCESSED_DIR / "corpus_clean.csv"

# Per-language splits
CYRILLIC_CSV = PROCESSED_DIR / "russian_posts.csv"
LATIN_CSV    = PROCESSED_DIR / "english_posts.csv"
GREEK_CSV    = PROCESSED_DIR / "greek_posts.csv"

# Lemmatized
CYRILLIC_LEMMATIZED_CSV = PROCESSED_DIR / "russian_posts_lemmatized.csv"
LATIN_LEMMATIZED_CSV    = PROCESSED_DIR / "english_posts_lemmatized.csv"
GREEK_LEMMATIZED_CSV    = PROCESSED_DIR / "greek_posts_lemmatized.csv"

# Frequency / n-gram
WORD_FREQ_CSV       = PROCESSED_DIR / "russian_word_frequency.csv"
BIGRAMS_CSV         = PROCESSED_DIR / "russian_bigrams.csv"
TRIGRAMS_CSV        = PROCESSED_DIR / "russian_trigrams.csv"
LATIN_WORD_FREQ_CSV = PROCESSED_DIR / "english_word_frequency.csv"
LATIN_BIGRAMS_CSV   = PROCESSED_DIR / "english_bigrams.csv"
LATIN_TRIGRAMS_CSV  = PROCESSED_DIR / "english_trigrams.csv"
GREEK_WORD_FREQ_CSV = PROCESSED_DIR / "greek_word_frequency.csv"
GREEK_BIGRAMS_CSV   = PROCESSED_DIR / "greek_bigrams.csv"
GREEK_TRIGRAMS_CSV  = PROCESSED_DIR / "greek_trigrams.csv"

# Twitter corpus
TWITTER_RAW_CSV       = PROCESSED_DIR / "twitter_kompromat_raw.csv"
TWITTER_PROCESSED_CSV = PROCESSED_DIR / "twitter_kompromat_processed.csv"

# ── Model directory (placeholder) ────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
