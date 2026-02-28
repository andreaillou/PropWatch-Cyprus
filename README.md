# PropWatch Cyprus

Computational pipeline for a two-paper academic project studying how Russian
foreign-broadcast propaganda amplifies pre-existing Cypriot social cleavages.

**Cleavage codes (Layer 2 annotation schema):** Cyprus division (`CY-DIV`),
EU scepticism (`EU-SKEP`), 2013 bail-in (`BAIL-IN`), Orthodox identity
(`ORTHO`), anti-elite populism (`ELIT`), migration (`MIGR`).

**Paper 1 — Propaganda techniques & narrative framing:**
SemEval-2020 14-class technique classification (XLM-RoBERTa-large) +
BERTopic narrative clustering with temporal analysis (H1, H4).

**Paper 2 — Amplification dynamics:**
Interrupted time series (ITS) around the January 2026 kompromat event;
`forwards` is the primary amplification proxy (H3).

Both papers share one corpus scraped from the Russian Embassy Cyprus
channel (`rusembcy`) on Telegram.

## Repository structure

```
├── main.ipynb                     # End-to-end pipeline notebook
├── requirements.txt               # Python dependencies (see sections)
├── .env.example                   # Template for API credentials
├── configs/
│   └── channels.yaml              # Telegram channel source list (by tier)
├── src/
│   ├── config.py                  # All file paths and scraping constants
│   ├── scraping/
│   │   ├── telegram.py            # Telethon scraper — collects message_id,
│   │   │                          #   views, forwards, reactions, reply_to_id,
│   │   │                          #   edit_date alongside date/channel/text
│   │   ├── twitter.py             # twarc2 scraper — Jan 2026 kompromat event
│   │   └── gdelt.py               # GDELT + Wayback scraper — Tier 1 archived
│   ├── preprocessing/
│   │   ├── filtering.py           # 4-step pipeline: length → spam → topic
│   │   │                          #   keywords → dedup; tags 9 binary columns
│   │   │                          #   (3 existing + 6 cleavage codes)
│   │   └── text_cleaning.py       # Text normalisation; lingua-py language
│   │                              #   detection (authoritative); script-type
│   │                              #   heuristic (secondary); splits corpus
│   │                              #   into russian / english / greek subsets
│   ├── analysis/
│   │   ├── lemmatization.py       # stanza-based lemmatization for Russian
│   │   │                          #   (ru pipeline) and Greek (el pipeline);
│   │   │                          #   spaCy is NOT used here
│   │   └── frequency.py           # Word frequency & n-gram analysis
│   │                              #   (script-agnostic, works on list[str])
│   └── classification/
│       └── model.py               # Propaganda classifier stub (TODO)
├── data/
│   ├── raw/
│   │   ├── telegram/              # Per-channel raw CSVs
│   │   └── twitter/               # Twitter raw CSVs
│   └── processed/                 # Merged corpus and analysis outputs
└── models/                        # Trained model weights (not committed)
```

## Quick-start

```bash
# 1. Clone and install
git clone <repo-url>
cd PropWatch-Cyprus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download stanza models (lemmatization — run once)
python -c "import stanza; stanza.download('ru'); stanza.download('el')"

# 3. Download spaCy English model (Track A syntactic features — run once)
python -m spacy download en_core_web_lg

# 4. Configure Telegram credentials
cp .env.example .env
# Edit .env — set TELEGRAM_APP_ID and TELEGRAM_API_HASH

# 5. Run the notebook
jupyter notebook main.ipynb
# — or run the scraper directly —
python -m src.scraping.telegram

# 6. (Optional) Twitter scraping — requires bearer token in .env
python -m src.scraping.twitter

# 7. (Optional) Tier 1 archived content — RT/Sputnik via GDELT + Wayback
python -m src.scraping.gdelt
# Note: fetch_text=True is slow (~1.5s per article). For metadata-only
# discovery pass, edit gdelt.py and set fetch_text=False first.
```

## Pipeline overview

```
scrape_channels()          → raw CSV (9 fields incl. forwards, reactions)
    ↓
filter_messages()          → length / spam / topic filter + 9 binary tags
    ↓
clean_and_split()          → text_cleaned | script_type | language
                             └─ russian_df / english_df / greek_df
    ↓
lemmatize_column()         → Russian lemmas (stanza ru)
lemmatize_greek_column()   → Greek lemmas  (stanza el)
    ↓
word_frequency()           → per-language top-N lemma frequency CSVs
compute_ngrams()           → bigrams / trigrams per language
    ↓
[TODO] BERTopic            → narrative clusters + temporal drift (H1, H4)
[TODO] XLM-RoBERTa-large  → SemEval-2020 14-class technique labels (Paper 1)
[TODO] ITS regression      → amplification analysis around Jan 2026 (H3)
```

## Status

| Component | Status |
| :-- | :-- |
| Telegram scraping (single channel) | ✅ Working |
| Multi-channel config (`channels.yaml`) | 🚧 Added — populate Tier 2 handles from source list |
| Twitter/X scraper (Jan 2026 kompromat) | 🚧 Added — requires bearer token |
| Tier 1 archived scraper (GDELT + Wayback + trafilatura) | 🚧 Added — `python -m src.scraping.gdelt` |
| Keyword filtering & cleavage-code tagging | ✅ Working |
| lingua-py language detection | ✅ Working |
| Text cleaning & corpus split | ✅ Working (bug fix applied) |
| Russian lemmatization (stanza) | ✅ Working |
| Greek lemmatization (stanza) | ✅ Working |
| Frequency / n-gram analysis | ✅ Working |
| BERTopic narrative clustering | 🚧 TODO — H1 / H4 |
| XLM-RoBERTa-large classification | 🚧 TODO — awaiting fine-tuned weights |
| Interrupted time series (H3) | 🚧 TODO — Jan 2026 kompromat event |
