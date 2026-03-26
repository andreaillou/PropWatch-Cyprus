"""Sitemap scraper for archived news sources."""

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import trafilatura
import requests

from src.config import RAW_DIR, START_DATE, ARCHIVED_RAW_DIR
from src.preprocessing.filtering import has_inclusion

logger = logging.getLogger(__name__)

_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# URL/title hint keywords used by the quick pre-filter.
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
    except Exception as exc:
        logger.warning("Failed to fetch sitemap %s: %s", url, exc)
        return None


def _collect_article_urls(
    sitemap_index_url: str,
    start_date: datetime = START_DATE,
    delay: float = 2.0,
) -> list[dict]:
    """Walk a sitemap index and return article URL metadata."""
    logger.info("Fetching sitemap index: %s", sitemap_index_url)
    root = _fetch_sitemap(sitemap_index_url, delay=delay)
    if root is None:
        return []

    study_years = {str(y) for y in range(start_date.year, datetime.now().year + 1)}
    article_entries: list[dict] = []

    # Root can be either sitemapindex or urlset.
    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if tag == "sitemapindex":
        child_urls = [
            loc.text.strip()
            for sm in root.findall("sm:sitemap", _NS)
            for loc in sm.findall("sm:loc", _NS)
            if loc.text
        ]
        # Fetch child sitemaps that appear to be in the study window.
        child_urls = [u for u in child_urls if any(yr in u for yr in study_years)]
        logger.info("Found %d child sitemaps in study window", len(child_urls))

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
        # Flat sitemap with direct URL entries.
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
    """Scrape one domain through its sitemap index."""
    ARCHIVED_RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Scraping %s", domain)
    all_entries = _collect_article_urls(sitemap_index_url, start_date, sitemap_delay)
    logger.info("Total URLs discovered: %d", len(all_entries))

    # Pre-filter is currently disabled.
    cyprus_entries = all_entries
    logger.debug(
        "Cyprus URL pre-filter disabled — processing all %d URLs",
        len(cyprus_entries),
    )

    rows: list[dict] = []
    for entry in cyprus_entries[:max_articles]:
        url = entry["url"]
        try:
            downloaded = trafilatura.fetch_url(url)
            text = (trafilatura.extract(downloaded) or "").strip()
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", url, exc)
            text = ""
            time.sleep(fetch_delay)
            continue

        # Fall back to sitemap title if extraction is empty.
        if not text:
            text = entry["title"]

        # Inclusion filter is currently disabled.
        if not text:
            time.sleep(fetch_delay * 0.5)
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
        logger.debug("Kept %d | %s", len(rows), url[:80])
        time.sleep(fetch_delay)

    df = pd.DataFrame(rows)
    out = ARCHIVED_RAW_DIR / f"{domain.replace('.', '_')}_raw.csv"
    df.to_csv(out, index=False)
    logger.info("Saved %d articles to %s", len(df), out)
    return df


def scrape_all_tier1(
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Scrape all tier1_archived sources from channels.yaml."""
    import yaml
    from src.config import ROOT_DIR

    cfg_path = config_path or ROOT_DIR / "configs" / "channels.yaml"
    with open(cfg_path) as fh:
        channel_config = yaml.safe_load(fh)

    all_frames: list[pd.DataFrame] = []

    for entry in channel_config.get("tier1_archived", []):
        sitemap_url = entry.get("sitemap_index")
        domain = entry.get("domain")
        if not sitemap_url or not domain:
            logger.debug(
                "No sitemap_index for %s — add it to channels.yaml",
                entry.get("label", "?"),
            )
            continue
        df = scrape_news_domain(domain=domain, sitemap_index_url=sitemap_url)
        all_frames.append(df)

    if not all_frames:
        logger.warning(
            "No data collected — check sitemap_index entries in channels.yaml",
        )
        return pd.DataFrame()

    merged = pd.concat(all_frames, ignore_index=True)
    merged_out = ARCHIVED_RAW_DIR / "tier1_archived_raw.csv"
    merged.to_csv(merged_out, index=False)
    logger.info("Done. Total %d articles saved to %s", len(merged), merged_out)
    return merged


if __name__ == "__main__":
    scrape_all_tier1()
