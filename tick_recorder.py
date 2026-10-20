"""
Record live ticks to daily CSV files (data/ticks_live/<SYMBOL>_<YYYY-MM-DD>.csv).

Deriv only serves ~1 day of tick history, so longer samples (e.g. to test the
Jump-index last-digit bias) have to be collected live. Places no trades.

  python tick_recorder.py                      # JD10 JD25 JD50 JD75
  python tick_recorder.py --symbols JD10 R_75
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient

OUT = Path("data/ticks_live")
STALE_SECS = 300


class Recorder:
    def __init__(self):
        self.files: dict[str, tuple[str, object]] = {}
        self.last_tick = time.time()
        self.counts: dict[str, int] = {}

    def _file(self, sym: str, epoch: int):
        day = datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d")
        cur = self.files.get(sym)
        if cur and cur[0] == day:
            return cur[1]
        if cur:
            cur[1].close()
        path = OUT / f"{sym}_{day}.csv"
        new = not path.exists()
        fh = path.open("a", buffering=1)
        if new:
            fh.write("epoch,quote,pip_size\n")
        self.files[sym] = (day, fh)
        return fh

    async def on_tick(self, tick: dict):
        sym, epoch = tick["symbol"], int(tick["epoch"])
        pip = tick.get("pip_size", "")
        # keep the exact displayed precision (a float would drop trailing zeros)
        q = f"{float(tick['quote']):.{int(pip)}f}" if pip != "" else str(tick["quote"])
        self._file(sym, epoch).write(f"{epoch},{q},{pip}\n")
        self.last_tick = time.time()
        self.counts[sym] = self.counts.get(sym, 0) + 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["JD10", "JD25", "JD50", "JD75"])
    a = ap.parse_args()

    load_dotenv()
    os.makedirs("logs", exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add("logs/tick_recorder_{time:YYYY-MM-DD}.log", rotation="00:00", retention="14 days", level="INFO",
               format="{time:HH:mm:ss} | {level:<7} | {message}",
               filter=lambda r: r["name"] != "src.api.client" or r["level"].no >= 30)
    logger.add(sys.stderr, level="WARNING")

    rec = Recorder()
    client = DerivClient(os.getenv("DERIV_API_TOKEN"), os.getenv("DERIV_APP_ID", "1089"))
    await client.connect()
    for s in a.symbols:
        client.on_tick(s, rec.on_tick)
        await client.subscribe_ticks(s)
    logger.info(f"Recording {', '.join(a.symbols)} -> {OUT}/")

    last_report = time.time()
    while True:
        await asyncio.sleep(30)
        if time.time() - rec.last_tick > STALE_SECS:
            logger.error(f"No ticks for {STALE_SECS}s — exiting so systemd restarts the recorder")
            sys.exit(1)
        if time.time() - last_report > 3600:
            logger.info("ticks recorded this session: " + ", ".join(f"{k}={v}" for k, v in sorted(rec.counts.items())))
            last_report = time.time()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
