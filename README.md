# Propaganda Paper Computation

Computational analysis of Russian-language propaganda dissemination through
Telegram channels

## Repository structure

```
├── main.ipynb                     # End-to-end notebook (scrape → analyse)
├── requirements.txt               # Python dependencies
├── .env.example                   # Template for API credentials
├── src/
│   ├── config.py                  # Paths, constants, env-var loading
│   ├── scraping/
│   │   └── telegram.py            # Telegram channel scraper (Telethon)
│   ├── preprocessing/
│   │   ├── filtering.py           # Keyword filtering & category tagging
│   │   └── text_cleaning.py       # Text normalisation & script detection
│   ├── analysis/
│   │   ├── lemmatization.py       # spaCy-based Russian lemmatizer
│   │   └── frequency.py           # Word frequency & n-gram analysis
│   └── classification/
│       └── model.py               # Propaganda classifier (draft / TODO)
├── cyprus_data/
│   └── telegram/russian_embassy/  # Scraped & processed data files
└── models/                        # Trained model weights (not committed)
```

## Quick-start

```bash
# 1. Clone and install
git clone <repo-url>
cd propaganda_paper_computation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download spaCy Russian model (needed for lemmatization)
python -m spacy download ru_core_news_lg

# 3. Configure Telegram credentials
cp .env.example .env
# Edit .env with your TELEGRAM_APP_ID and TELEGRAM_API_HASH

# 4. Run the notebook or individual modules
jupyter notebook main.ipynb
# — or —
python -m src.scraping.telegram
```

## Status

| Component                       | Status                                          |
| ------------------------------- | ----------------------------------------------- |
| Telegram scraping               | ✅ Working                                      |
| Keyword filtering & tagging     | ✅ Working                                      |
| Text cleaning & script split    | ✅ Working                                      |
| Russian lemmatization           | ✅ Working                                      |
| Frequency / n-gram analysis     | ✅ Working                                      |
| Propaganda classification model | 🚧 Draft — awaiting trained model & full corpus |
