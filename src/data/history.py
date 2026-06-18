"""
Fetches historical tick data from the Deriv ticks_history API.

Ticks are cached to data/<symbol>_<count>.json so you don't re-fetch on every run.
Pass --fresh to backtest.py to force a new download.
"""

import asyncio
import json
import os
from pathlib import Path

import websockets
from loguru import logger

CACHE_DIR = Path("data")


def _cache_path(symbol: str, count: int) -> Path:
    return CACHE_DIR / f"{symbol}_{count}.json"


async def _fetch_from_api(symbol: str, count: int) -> list[dict]:
    """
    Opens a one-shot WebSocket connection and pulls tick history.
    ticks_history for synthetic indices is public — no auth required.
    Uses app_id 1089 (Deriv public demo app).
    """
    url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    logger.info(f"Connecting to Deriv API to fetch {count} ticks for {symbol}...")

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "ticks",
            "req_id": 1,
        }))

        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)

        if msg.get("error"):
            raise RuntimeError(msg["error"]["message"])

        history = msg["history"]
        prices = history["prices"]
        times = history["times"]

    ticks = [
        {"epoch": times[i], "symbol": symbol, "quote": str(prices[i])}
        for i in range(len(prices))
    ]
    logger.info(f"Fetched {len(ticks)} ticks (earliest: epoch {times[0]}, latest: epoch {times[-1]})")
    return ticks


def fetch_ticks(
    symbol: str,
    count: int = 5000,
    fresh: bool = False,
    **_kwargs,  # absorb unused app_id / api_token args for backwards compat
) -> list[dict]:
    """
    Returns `count` historical ticks for `symbol`.
    Loads from cache if available; set fresh=True to re-download.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(symbol, count)

    if path.exists() and not fresh:
        logger.info(f"Loading {count} ticks from cache: {path}")
        with open(path) as f:
            return json.load(f)

    ticks = asyncio.run(_fetch_from_api(symbol, count))

    with open(path, "w") as f:
        json.dump(ticks, f)
    logger.info(f"Cached to {path}")
    return ticks
