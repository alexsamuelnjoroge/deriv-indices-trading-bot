"""
Fetches historical tick data from the Deriv ticks_history API.

The API caps at 5,000 ticks per request. fetch_ticks() transparently paginates
by walking backwards through time until the requested total is reached.

Ticks are cached to data/<symbol>_<count>.json so you don't re-fetch on every run.
Pass fresh=True to force a new download.
"""

import asyncio
import json
from pathlib import Path

import websockets
from loguru import logger

CACHE_DIR = Path("data")
API_BATCH  = 5000   # Deriv hard limit per ticks_history call


def _cache_path(symbol: str, count: int) -> Path:
    return CACHE_DIR / f"{symbol}_{count}.json"


async def _fetch_batch(ws, symbol: str, count: int, end) -> list[dict]:
    """Single ticks_history call; end is 'latest' or an integer epoch."""
    payload = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": min(count, API_BATCH),
        "end": end,
        "style": "ticks",
        "req_id": 1,
    }
    await ws.send(json.dumps(payload))
    raw = await asyncio.wait_for(ws.recv(), timeout=30)
    msg = json.loads(raw)

    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])

    prices = msg["history"]["prices"]
    times  = msg["history"]["times"]
    return [
        {"epoch": times[i], "symbol": symbol, "quote": str(prices[i])}
        for i in range(len(prices))
    ]


async def _fetch_paginated(symbol: str, total: int) -> list[dict]:
    """
    Fetches `total` ticks by making as many API calls as needed (each ≤ 5000).
    Walks backwards in time: first call gets the most-recent ticks, each
    subsequent call ends just before where the previous one started.
    """
    url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    all_ticks: list[dict] = []
    end = "latest"

    async with websockets.connect(url) as ws:
        while len(all_ticks) < total:
            need  = min(API_BATCH, total - len(all_ticks))
            batch = await _fetch_batch(ws, symbol, need, end)

            if not batch:
                logger.warning("Empty batch returned — no more history available.")
                break

            all_ticks = batch + all_ticks   # prepend so result is chronological

            logger.info(
                f"Fetched {len(batch)} ticks (epoch {batch[0]['epoch']} "
                f"to {batch[-1]['epoch']}) | total so far: {len(all_ticks)}"
            )

            if len(batch) < need:
                logger.warning("API returned fewer ticks than requested — history exhausted.")
                break

            # Move end pointer back one tick for the next call
            end = batch[0]["epoch"] - 1

    logger.info(f"Pagination complete: {len(all_ticks)} ticks fetched for {symbol}")
    return all_ticks


def fetch_ticks(
    symbol: str,
    count: int = 5000,
    fresh: bool = False,
    **_kwargs,
) -> list[dict]:
    """
    Returns up to `count` historical ticks for `symbol`.
    Loads from cache if available; set fresh=True to re-download.
    Automatically paginates when count > 5000.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(symbol, count)

    if path.exists() and not fresh:
        logger.info(f"Loading {count} ticks from cache: {path}")
        with open(path) as f:
            return json.load(f)

    logger.info(f"Fetching {count} ticks for {symbol} (may require {(count - 1) // API_BATCH + 1} API calls)...")
    ticks = asyncio.run(_fetch_paginated(symbol, count))

    with open(path, "w") as f:
        json.dump(ticks, f)
    logger.info(f"Cached {len(ticks)} ticks to {path}")
    return ticks
