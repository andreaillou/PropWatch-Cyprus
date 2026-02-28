"""Direct sitemap scraper for Tier 1 news sources (RT, Sputnik).

Discovers article URLs from XML sitemaps, filters for Cyprus relevance
using the existing inclusion keyword list, then fetches full text via
trafilatura. No external API required — works with any accessible domain.

Strategy:
  1. Fetch sitemap index → collect per-month child sitemaps
  2. Filter child sitemaps to the study window (START_DATE → today)
  3. Parse each child sitemap → collect article URLs + lastmod dates
  4. Pre-filter URL/title by Cyprus keywords to avoid fetching every article
  5. Fetch full text via trafilatura for matched URLs
  6. Save per-domain CSVs + merged output

Usage::

    python -m src.scraping.news
"""

import hashlib
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import trafilatura
import requests

from src.config import RAW_DIR, START_DATE, ARCHIVED_RAW_DIR
from src.preprocessing.filtering import has_inclusion

# ── Sitemap namespace ─────────────────────────────────────────────────────────
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# ── Cyprus URL-level pre-filter keywords (ASCII only, lowercase) ──────────────
# These are checked against the article URL and sitemap <title> only —
# cheap pre-filter before the expensive trafilatura fetch.
# Kept intentionally broad; has_inclusion() does the precise filtering.
_CYPRUS_URL_HINTS = [
    "cyprus", "nicosia", "famagusta", "christodoulides",
    "limassol", "larnaca", "paphos",
]


def _is_cyprus_hint(url: str, title: str = "") -> bool:
    """Quick pre-screen: does the URL or title mention Cyprus keywords?"""
    combined = (url + " " + title).lower()
    return any(hint in combined for hint in _CYPRUS_URL_HINTS)


def _fetch_sitemap(url: str, delay: float = 2.0) -> ET.Element | None:
    """Fetch and parse one XML sitemap. Returns root element or None."""
    time.sleep(delay)
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return ET.fromstring(resp.content)
    except Exception as e:
        print(f"   [!] Failed to fetch sitemap {url}: {e}")
        return None


def _collect_article_urls(
    sitemap_index_url: str,
    start_date: datetime = START_DATE,
    delay: float = 2.0,
) -> list[dict]:
    """Walk a sitemap index and return article dicts within the date window.

    Each dict has: ``url``, ``lastmod`` (str), ``title`` (str, may be empty).

    Only child sitemaps whose name suggests they fall within the study
    window are fetched (cheap heuristic: year appears in the sitemap URL).
    """
    print(f"   Fetching sitemap index: {sitemap_index_url}")
    root = _fetch_sitemap(sitemap_index_url, delay=delay)
    if root is None:
        return []

    study_years = {str(y) for y in range(start_date.year, datetime.now().year + 1)}
    article_entries: list[dict] = []

    # Root may be a <sitemapindex> (list of child sitemaps) or directly
    # a <urlset> (flat list of URLs — some sites skip the index level)
    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if tag == "sitemapindex":
        child_urls = [
            loc.text.strip()
            for sm in root.findall("sm:sitemap", _NS)
            for loc in sm.findall("sm:loc", _NS)
            if loc.text
        ]
        # Heuristic: only fetch child sitemaps that contain a study year in URL
        child_urls = [u for u in child_urls if any(yr in u for yr in study_years)]
        print(f"   Found {len(child_urls)} child sitemaps in study window")

        for child_url in child_urls:
            child_root = _fetch_sitemap(child_url, delay=delay)
            if child_root is None:
                continue
            for url_el in child_root.findall("sm:url", _NS):
                loc = url_el.findtext("sm:loc", namespaces=_NS) or ""
                lastmod = url_el.findtext("sm:lastmod", namespaces=_NS) or ""
                title = url_el.findtext(
                    "sm:news/sm:title",
                    namespaces={**_NS, "sm": "http://www.google.com/schemas/sitemap-news/0.9"}
                ) or ""
                if loc:
                    article_entries.append({"url": loc, "lastmod": lastmod, "title": title})

    elif tag == "urlset":
        # Flat sitemap — no child level
        for url_el in root.findall("sm:url", _NS):
            loc = url_el.findtext("sm:loc", namespaces=_NS) or ""
            lastmod = url_el.findtext("sm:lastmod", namespaces=_NS) or ""
            if loc:
                article_entries.append({"url": loc, "lastmod": lastmod, "title": ""})

    return article_entries


def scrape_news_domain(
    domain: str,
    sitemap_index_url: str,
    region: str = "tier1_archived",
    start_date: datetime = START_DATE,
    fetch_delay: float = 1.5,
    sitemap_delay: float = 2.0,
    max_articles: int = 500,
) -> pd.DataFrame:
    """Scrape one news domain via its sitemap index.

    Parameters
    ----------
    domain            : e.g. ``"rt.com"`` — used as the ``channel`` field
    sitemap_index_url : URL of the top-level sitemap index XML
    region            : corpus region label (``"tier1_archived"``)
    start_date        : only keep articles on or after this date
    fetch_delay       : seconds between trafilatura article fetches
    sitemap_delay     : seconds between sitemap XML requests
    max_articles      : safety cap — stop after this many fetched articles

    Returns
    -------
    pd.DataFrame in the standard corpus schema
    """
    ARCHIVED_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Scraping {domain} ===")
    all_entries = _collect_article_urls(sitemap_index_url, start_date, sitemap_delay)
    print(f"   Total URLs discovered: {len(all_entries)}")

    # Pre-filter by Cyprus URL/title hints
    cyprus_entries = [e for e in all_entries if _is_cyprus_hint(e["url"], e["title"])]
    print(f"   After Cyprus URL pre-filter: {len(cyprus_entries)}")

    rows: list[dict] = []
    for entry in cyprus_entries[:max_articles]:
        url = entry["url"]
        try:
            downloaded = trafilatura.fetch_url(url)
            text = (trafilatura.extract(downloaded) or "").strip()
        except Exception as e:
            print(f"   [!] Fetch failed for {url}: {e}")
            text = ""
            time.sleep(fetch_delay)
            continue

        # Fall back to title if trafilatura returned nothing
        if not text:
            text = entry["title"]

        # Skip if text still doesn't pass the full inclusion filter
        if not has_inclusion(text):
            time.sleep(fetch_delay * 0.5)  # shorter sleep for skipped articles
            continue

        msg_id = hashlib.md5(url.encode()).hexdigest()
        rows.append({
            "message_id":  msg_id,
            "date":        entry["lastmod"],
            "channel":     domain,
            "region":      region,
            "text":        text,
            "views":       0,
            "forwards":    0,
            "reactions":   0,
            "reply_to_id": None,
            "edit_date":   None,
            "source_url":  url,
        })
        print(f"   ✓ {len(rows)} kept | {url[:80]}")
        time.sleep(fetch_delay)

    df = pd.DataFrame(rows)
    out = ARCHIVED_RAW_DIR / f"{domain.replace('.', '_')}_raw.csv"
    df.to_csv(out, index=False)
    print(f"   Saved {len(df)} articles to {out}")
    return df


def scrape_all_tier1(
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Scrape all Tier 1 sources defined in configs/channels.yaml.

    Reads ``sitemap_index`` URL from each ``tier1_archived`` entry.
    Skips entries without a ``sitemap_index`` key.
    """
    import yaml
    from src.config import ROOT_DIR

    cfg_path = config_path or ROOT_DIR / "configs" / "channels.yaml"
    with open(cfg_path) as f:
        data = yaml.safe_load(f)

    all_frames: list[pd.DataFrame] = []

    for entry in data.get("tier1_archived", []):
        sitemap_url = entry.get("sitemap_index")
        domain = entry.get("domain")
        if not sitemap_url or not domain:
            print(f"   [skip] No sitemap_index for {entry.get('label', '?')} — add it to channels.yaml")
            continue
        df = scrape_news_domain(domain=domain, sitemap_index_url=sitemap_url)
        all_frames.append(df)

    if not all_frames:
        print("No data collected — check sitemap_index entries in channels.yaml")
        return pd.DataFrame()

    merged = pd.concat(all_frames, ignore_index=True)
    merged_out = ARCHIVED_RAW_DIR / "tier1_archived_raw.csv"
    merged.to_csv(merged_out, index=False)
    print(f"\nDone. Total {len(merged)} articles saved to {merged_out}")
    return merged


if __name__ == "__main__":
    scrape_all_tier1()
