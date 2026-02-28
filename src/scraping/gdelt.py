"""GDELT Doc 2.0 scraper for Tier 1 archived sources (RT, Sputnik).

Queries the GDELT Document API for articles from rt.com and
sputniknews.com mentioning Cyprus-relevant keywords. Then fetches
full article text from the Internet Archive (Wayback Machine) via
trafilatura, falling back to the live URL if the archive returns nothing.

No authentication required. Rate-limit: max 1 req/sec to GDELT,
be polite to Archive.org.

Usage::

    python -m src.scraping.gdelt
"""

import hashlib
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import trafilatura

from src.config import RAW_DIR, START_DATE

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Domains to target — foreign-broadcast only (not domestic Russian)
TIER1_DOMAINS = ["rt.com", "sputniknews.com", "sputnik.md"]

# Cyprus-relevant keyword query sent to GDELT
# Keep broad — downstream filtering handles specificity
CYPRUS_QUERY = (
    '"Cyprus" OR "Κύπρος" OR "Кипр" OR '
    '"Christodoulides" OR "rusembcy" OR "ELAM"'
)

# GDELT date range: study window start → today
GDELT_START = START_DATE.strftime("%Y%m%d%H%M%S")  # e.g. "20240101000000"

ARCHIVED_RAW_DIR = RAW_DIR / "archived"


def _gdelt_query(
    domain: str,
    query: str = CYPRUS_QUERY,
    start: str = GDELT_START,
    max_records: int = 250,
) -> list[dict]:
    """Call GDELT Doc API and return article metadata for one domain.

    Parameters
    ----------
    domain : e.g. ``"rt.com"``
    query  : free-text query string
    start  : GDELT datetime string ``"YYYYMMDDHHMMSS"``
    max_records : capped at 250 by GDELT free tier

    Returns list of dicts with keys: url, title, seendate, domain, language, sourcecountry
    """
    params = {
        "query": f"{query} domain:{domain}",
        "mode": "artlist",
        "maxrecords": max_records,
        "startdatetime": start,
        "format": "json",
        "sort": "DateDesc",
    }
    resp = requests.get(GDELT_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])


def _fetch_text(url: str, try_archive: bool = True) -> str:
    """Fetch full article text via trafilatura.

    First attempts the Wayback Machine snapshot (if try_archive=True),
    then falls back to the live URL. Returns empty string on failure.
    """
    if try_archive:
        archive_url = f"https://web.archive.org/web/{url}"
        try:
            downloaded = trafilatura.fetch_url(archive_url)
            text = trafilatura.extract(downloaded) or ""
            if text.strip():
                return text.strip()
        except Exception:
            pass

    # Fallback: live URL (may be blocked from EU IP)
    try:
        downloaded = trafilatura.fetch_url(url)
        return (trafilatura.extract(downloaded) or "").strip()
    except Exception:
        return ""


def scrape_archived(
    domains: list[str] = TIER1_DOMAINS,
    query: str = CYPRUS_QUERY,
    start: str = GDELT_START,
    max_records: int = 250,
    fetch_text: bool = True,
    delay: float = 1.5,
) -> pd.DataFrame:
    """Collect archived articles from Tier 1 sources via GDELT + Wayback.

    Parameters
    ----------
    domains     : list of domains to query
    query       : GDELT keyword query
    start       : GDELT start datetime string
    max_records : per-domain article cap (GDELT free tier max: 250)
    fetch_text  : if True, fetches full article text via trafilatura;
                  if False, returns metadata only (much faster)
    delay       : seconds to sleep between requests (be polite)

    Returns
    -------
    pd.DataFrame with the standard corpus schema:
    message_id, date, channel, region, text, views, forwards,
    reactions, reply_to_id, edit_date
    """
    ARCHIVED_RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for domain in domains:
        print(f"--- Querying GDELT for: {domain} ---")
        try:
            articles = _gdelt_query(domain, query=query, start=start, max_records=max_records)
            print(f"   Found {len(articles)} articles")
        except Exception as e:
            print(f"   [!] GDELT query failed for {domain}: {e}")
            continue

        for article in articles:
            url   = article.get("url", "")
            title = article.get("title", "")
            date  = article.get("seendate", "")

            text = ""
            if fetch_text and url:
                text = _fetch_text(url)
                time.sleep(delay)

            # Use title as text fallback if full fetch failed
            if not text:
                text = title

            # Stable message_id: MD5 of URL (no Telegram IDs available)
            msg_id = hashlib.md5(url.encode()).hexdigest() if url else ""

            rows.append({
                "message_id":  msg_id,
                "date":        date,
                "channel":     domain,
                "region":      "tier1_archived",
                "text":        text,
                "views":       0,
                "forwards":    0,
                "reactions":   0,
                "reply_to_id": None,
                "edit_date":   None,
                # Preserve source URL for citation/traceability — not in
                # the Telegram schema but needed for legal/ethics documentation
                "source_url":  url,
            })

        # Save per-domain immediately (crash resilience)
        domain_df = pd.DataFrame([r for r in rows if r["channel"] == domain])
        out = ARCHIVED_RAW_DIR / f"{domain.replace('.', '_')}_raw.csv"
        domain_df.to_csv(out, index=False)
        print(f"   Saved {len(domain_df)} rows to {out}")

        time.sleep(delay)

    df = pd.DataFrame(rows)
    merged_out = ARCHIVED_RAW_DIR / "tier1_archived_raw.csv"
    df.to_csv(merged_out, index=False)
    print(f"\nDone. Saved {len(df)} total rows to {merged_out}")
    return df


if __name__ == "__main__":
    scrape_archived()
