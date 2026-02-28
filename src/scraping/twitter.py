"""Twitter/X scraper for the January 2026 Cyprus kompromat event.

Uses twarc2 (Twitter API v2 recent search endpoint).
Set TWITTER_BEARER_TOKEN in .env before running.

Usage::

    python -m src.scraping.twitter
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from twarc import Twarc2, expansions

from src.config import TWITTER_BEARER_TOKEN, TWITTER_RAW_DIR, TWITTER_RAW_CSV

# Default query for the Jan 2026 Christodoulides kompromat operation.
# Expand or narrow via CLI args or by passing query= to scrape_twitter().
DEFAULT_QUERY = (
    "(Christodoulides OR #Cyprus OR Κυριάκος) "
    "lang:en OR lang:ru OR lang:el "
    "-is:retweet"
)

DEFAULT_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
DEFAULT_END   = datetime(2026, 2, 1, tzinfo=timezone.utc)


def scrape_twitter(
    query: str = DEFAULT_QUERY,
    start_time: datetime = DEFAULT_START,
    end_time: datetime = DEFAULT_END,
    max_results: int = 100,          # per page; twarc2 handles pagination
    output_path: Path = TWITTER_RAW_CSV,
) -> pd.DataFrame:
    """Collect tweets matching *query* within the given date window.

    Parameters match the twarc2 recent search endpoint. For full-archive
    access (pre-7-day window), an Academic Research or Pro tier token is
    required — update client initialisation accordingly.
    """
    client = Twarc2(bearer_token=TWITTER_BEARER_TOKEN)
    rows: list[dict] = []

    for page in client.search_recent(
        query=query,
        start_time=start_time,
        end_time=end_time,
        max_results=max_results,
    ):
        result = expansions.flatten(page)
        for tweet in result:
            metrics = tweet.get("public_metrics", {})
            refs    = tweet.get("referenced_tweets", [])
            reply_id = next(
                (r["id"] for r in refs if r["type"] == "replied_to"), None
            )
            rows.append({
                "message_id":  tweet["id"],
                "date":        tweet["created_at"],
                "channel":     "twitter",
                "source":      tweet.get("author", {}).get("username", ""),
                "text":        tweet["text"],
                "views":       0,
                "forwards":    metrics.get("retweet_count", 0),
                "reactions":   metrics.get("like_count", 0),
                "reply_to_id": reply_id,
                "edit_date":   None,
            })
        time.sleep(1)  # stay inside rate window

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} tweets to {output_path}")
    return df


if __name__ == "__main__":
    scrape_twitter()
