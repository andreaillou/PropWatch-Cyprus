![Status: Active Research](https://img.shields.io/badge/status-active%20research-blue) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-brightgreen)

# PropWatch Cyprus

Computational pipeline for a two-paper academic project studying how Russian
foreign-broadcast propaganda amplifies pre-existing Cypriot social cleavages.

Part of a two-paper academic publication project on computational propaganda analysis.

**Cleavage codes (Layer 2 annotation schema):** Cyprus division (`CY-DIV`),
EU scepticism (`EU-SKEP`), 2013 bail-in (`BAIL-IN`), Orthodox identity
(`ORTHO`), anti-elite populism (`ELIT`), migration (`MIGR`).

**Paper 1 — Propaganda techniques & narrative framing:**
SemEval-2020 14-class technique classification (XLM-RoBERTa-large) +
BERTopic narrative clustering with temporal analysis (H1, H4).

**Paper 2 — Amplification dynamics:**
Interrupted time series (ITS) around the January 2026 kompromat event;
`forwards` is the primary amplification proxy (H3).

Both papers share a multilingual corpus drawn from three Telegram channels
(Russian Embassy Cyprus, Rybar, War on Fakes), Tier 1 archived news sites
(RT, Sputnik, Vergina TV via sitemap + trafilatura), and Twitter/X.

## Repository structure

```
├── main.ipynb                     # End-to-end pipeline notebook
├── requirements.txt               # Python dependencies (see sections)
├── LICENSE                        # MIT license
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
│   │   └── news.py                # Direct sitemap scraper — Tier 1 archived
│   ├── preprocessing/
│   │   ├── filtering.py           # 5-step pipeline: length → nav-dump → spam
│   │   │                          #   → NLI relevance gate → dedup; then
│   │   │                          #   tag_categories adds 9 binary keyword
│   │   │                          #   columns (WAR, KEY_ACT, IDEAL_TER +
│   │   │                          #   6 cleavage codes)
│   │   └── text_cleaning.py       # Text normalisation; lingua-py language
│   │                              #   detection (authoritative); script-type
│   │                              #   heuristic (secondary); splits corpus
│   │                              #   into russian / english / greek subsets
│   ├── analysis/
│   │   ├── lemmatization.py       # Russian (stanza ru) + Greek (stanza el) +
│   │   │                          #   English (spaCy en_core_web_lg);
│   │   │                          #   incremental — skips already-lemmatized rows
│   │   └── frequency.py           # Word frequency & n-gram analysis
│   │                              #   (script-agnostic, works on list[str])
│   └── classification/
│       └── model.py               # XLM-RoBERTa-large 14-class classifier
│                                  #   (SemEval-2020 schema; skeleton only)
├── data/
│   ├── raw/
│   │   ├── archived/              # Tier 1 news CSVs (RT, Sputnik, Vergina TV)
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

# 7. Tier 1 archived content — RT/Sputnik via direct sitemap
python -m src.scraping.news
# Add sitemap_index URLs to configs/channels.yaml for each source before running.
# RT English and RT Russian are pre-configured. Sputnik URLs need verification.
```

## Pipeline overview

```
scrape_channels()            → raw CSV (10 fields incl. forwards, reactions)
scrape_all_tier1()           → archived news via sitemap + trafilatura
scrape_twitter()             → Twitter/X via twarc2
    ↓
merge & concat               → unified corpus_raw.csv
    ↓
filter_messages()            → length → nav-dump → spam → NLI gate → dedup
tag_categories()             → 9 binary keyword columns (no rows removed)
    ↓
clean_and_split()            → text_cleaned | script_type | language
                               └─ russian_df / english_df / greek_df
    ↓
lemmatize_column()           → Russian lemmas  (stanza ru)
lemmatize_greek_column()     → Greek lemmas    (stanza el)
lemmatize_english_column()   → English lemmas  (spaCy en_core_web_lg)
    ↓
word_frequency()             → per-language top-N lemma frequency CSVs
compute_ngrams()             → bigrams / trigrams per language
    ↓
[planned] BERTopic           → narrative clusters + temporal drift (H1, H4)
[planned] XLM-RoBERTa        → SemEval-2020 14-class technique labels (Paper 1)
[planned] ITS regression      → amplification analysis around Jan 2026 (H3)
```

## Status

| Component                                                  | Status                                   |
| :--------------------------------------------------------- | :--------------------------------------- |
| Telegram scraping (multi-channel)                          | ✅ Working                               |
| Multi-channel config (`channels.yaml`)                     | ✅ Working                               |
| Tier 1 news scraper (RT/Sputnik via sitemap + trafilatura) | ✅ Working                               |
| Keyword filtering & cleavage-code tagging                  | ✅ Working                               |
| Zero-shot NLI political-relevance gate                     | ✅ Working                               |
| lingua-py language detection                               | ✅ Working                               |
| Text cleaning & corpus split                               | ✅ Working                               |
| Russian lemmatization (stanza)                             | ✅ Working                               |
| Greek lemmatization (stanza)                               | ✅ Working                               |
| English lemmatization (spaCy)                              | ✅ Working                               |
| Frequency / n-gram analysis                                | ✅ Working                               |
| BERTopic narrative clustering                              | 🚧 Planned — H1 / H4                     |
| XLM-RoBERTa-large classification                           | 🚧 Planned — awaiting fine-tuned weights |
| Interrupted time series (H3)                               | 🚧 Planned — Jan 2026 kompromat event    |

## License

This repository (code, notebooks, data, and all other materials) is
licensed under the **Academic Research License (ARL-1.0)**.  
Use by commercial entities or government bodies (and their agents) is
**explicitly prohibited**. See [LICENSE](./LICENSE) for full terms.

## Authors

Andrey Vyalkov — pipeline architecture & implementation | Iuliia Shirina — research design & annotations
