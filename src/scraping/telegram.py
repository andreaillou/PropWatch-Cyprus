"""Telegram channel scraper using Telethon."""

import asyncio
import logging
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
    TELEGRAM_RAW_DIR,
)

logger = logging.getLogger(__name__)


async def scrape_channels(
    api_id: int = TELEGRAM_APP_ID,
    api_hash: str = TELEGRAM_API_HASH,
    channels: dict[str, list[str]] = CHANNELS,
    start_date=START_DATE,
    message_limit: int = MESSAGE_LIMIT,
    output_path=RAW_CSV,
) -> pd.DataFrame:
    """Scrape messages from the configured Telegram channels."""
    all_data: list[dict] = []

    async with TelegramClient("anon", api_id, api_hash) as client:
        for region, names in channels.items():
            for name in names:
                logger.info("Processing channel: %s", name)
                try:
                    entity = await client.get_entity(name)
                    msg_count = 0
                    channel_data: list[dict] = []

                    channel_dir = TELEGRAM_RAW_DIR / name
                    channel_dir.mkdir(parents=True, exist_ok=True)

                    async for msg in client.iter_messages(entity, limit=message_limit):
                        if msg.date < start_date:
                            break
                        if msg.text:
                            row = {
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
                            channel_data.append(row)
                            msg_count += 1

                    # Save per-channel output.
                    if channel_data:
                        pd.DataFrame(channel_data).to_csv(
                            channel_dir / f"{name}_raw.csv", index=False
                        )
                    all_data.extend(channel_data)

                    logger.info(
                        "Scraped %d messages from %s", msg_count, name,
                    )

                except FloodWaitError as exc:
                    logger.warning(
                        "Rate limited by Telegram. Waiting %d seconds.",
                        exc.seconds,
                    )
                    await asyncio.sleep(exc.seconds + 5)

                except Exception as exc:
                    logger.error(
                        "Error scraping %s: %s", name, exc,
                    )

                delay = random.randint(10, 25)
                logger.debug("Pause: %ds", delay)
                await asyncio.sleep(delay)

    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    logger.info("Done. Saved %d rows to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    asyncio.run(scrape_channels())
