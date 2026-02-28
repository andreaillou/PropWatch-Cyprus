"""Telegram channel scraper using Telethon.

Usage (CLI)::

    python -m src.scraping.telegram

Or import and call :func:`scrape_channels` from a notebook / script.
"""

import asyncio
import random

import pandas as pd
from telethon.sync import TelegramClient
from telethon.errors import FloodWaitError

from src.config import (
    TELEGRAM_APP_ID,
    TELEGRAM_API_HASH,
    CHANNELS,
    START_DATE,
    MESSAGE_LIMIT,
    RAW_CSV,
)


async def scrape_channels(
    api_id: int = TELEGRAM_APP_ID,
    api_hash: str = TELEGRAM_API_HASH,
    channels: dict[str, list[str]] = CHANNELS,
    start_date=START_DATE,
    message_limit: int = MESSAGE_LIMIT,
    output_path=RAW_CSV,
) -> pd.DataFrame:
    """Scrape messages from the configured Telegram channels.

    Parameters
    ----------
    api_id, api_hash : Telegram API credentials.
    channels : ``{region: [channel_name, ...]}`` mapping.
    start_date : Only keep messages *after* this date.
    message_limit : Max messages to fetch per channel.
    output_path : Where to save the resulting CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``message_id``  – Telegram message ID; use for deduplication and
          span-to-source linking in annotation.
        - ``date``        – UTC timestamp of the original message.
        - ``channel``     – Channel username/handle.
        - ``region``      – Region label from the ``CHANNELS`` mapping.
        - ``text``        – Message text (newlines collapsed to spaces).
        - ``views``       – View count at scrape time (engagement signal).
        - ``forwards``    – Forward count; primary amplification proxy for
          H3 interrupted time-series analysis.  **Cannot be recovered
          post-hoc — must be collected at scrape time.**
        - ``reactions``   – Total reaction count (sum across all emoji types).
        - ``reply_to_id`` – ``message_id`` of the parent message if this is
          a reply, else ``None``.
        - ``edit_date``   – UTC timestamp of the last edit, or ``None``.
    """
    all_data: list[dict] = []

    async with TelegramClient("anon", api_id, api_hash) as client:
        for region, names in channels.items():
            for name in names:
                print(f"--- Processing: {name} ---")
                try:
                    entity = await client.get_entity(name)
                    msg_count = 0

                    async for msg in client.iter_messages(entity, limit=message_limit):
                        if msg.date < start_date:
                            break
                        if msg.text:
                            all_data.append(
                                {
                                    "message_id": msg.id,
                                    "date": msg.date,
                                    "channel": name,
                                    "region": region,
                                    "text": msg.text.replace("\n", " "),
                                    "views": msg.views or 0,
                                    "forwards": msg.forwards or 0,
                                    "reactions": sum(r.count for r in msg.reactions.results) if msg.reactions else 0,
                                    "reply_to_id": msg.reply_to.reply_to_msg_id if msg.reply_to else None,
                                    "edit_date": msg.edit_date,
                                }
                            )
                            msg_count += 1

                    print(f"   Success: Scraped {msg_count} messages.")

                except FloodWaitError as e:
                    print(
                        f"   [!] Rate Limited by Telegram. "
                        f"Must wait {e.seconds} seconds."
                    )
                    await asyncio.sleep(e.seconds + 5)

                except Exception as e:
                    print(f"   [!] General Error on {name}: {e}")

                delay = random.randint(10, 25)
                print(f"   [Anti-Ban] Pausing for {delay}s...")
                await asyncio.sleep(delay)

    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    print(f"\nDone! Saved {len(df)} rows to {output_path}")
    return df


# Allow running as ``python -m src.scraping.telegram``
if __name__ == "__main__":
    asyncio.run(scrape_channels())
