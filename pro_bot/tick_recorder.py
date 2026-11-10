"""
MT5 tick recorder — MT5 keeps only ~1 day of tick history, so order-flow research
(bid/ask update imbalance, quote intensity) needs ticks recorded going forward.

    python -m pro_bot.tick_recorder [--symbols XAUUSD,XAGUSD,EURUSD]

Writes data/ticks/<SYMBOL>/<UTC date>.csv.gz with columns utc_msc,bid,ask,flags.
Resumes from data/ticks/_state.json after a restart (gap = whatever MT5 still holds).
"""

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

import MetaTrader5 as mt5

from pro_bot.execution.mt5_client import MT5Client
from pro_bot.incubator import server_offset

ROOT = Path("data/ticks")
STATE = ROOT / "_state.json"
POLL_SECONDS = 30


def server_dt(server_ts: float) -> datetime:
    """MT5 tick functions take server time expressed as a UTC datetime."""
    return datetime.fromtimestamp(server_ts, timezone.utc)


def write_ticks(sym: str, ticks) -> None:
    utc_msc = ticks["time_msc"].astype(np.int64) - server_offset(int(ticks["time"][-1])) * 1000
    day = utc_msc // 86_400_000
    for d in np.unique(day):
        m = day == d
        f = ROOT / sym / f"{datetime.fromtimestamp(int(d) * 86400, timezone.utc):%Y-%m-%d}.csv.gz"
        f.parent.mkdir(parents=True, exist_ok=True)
        new = not f.exists()
        lines = "".join(f"{u},{b},{a},{fl}\n" for u, b, a, fl in
                        zip(utc_msc[m], ticks["bid"][m], ticks["ask"][m], ticks["flags"][m]))
        with gzip.open(f, "at") as fh:
            if new:
                fh.write("utc_msc,bid,ask,flags\n")
            fh.write(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="XAUUSD,XAGUSD,EURUSD")
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    load_dotenv()
    Path("logs").mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level:<7} | {message}")
    logger.add("logs/tick_recorder_{time:YYYYMMDD}.log", level="INFO", rotation="1 day", retention="30 days")

    if not MT5Client().connect():
        sys.exit("MT5 connection failed")
    for s in symbols:
        mt5.symbol_select(s, True)

    ROOT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    totals = {s: 0 for s in symbols}
    last_report = time.time()
    logger.info(f"tick recorder up | symbols={symbols} | resume={state}")

    while True:
        try:
            now_srv = time.time() + server_offset(time.time())
            for sym in symbols:
                last = state.get(sym, int((now_srv - 86400) * 1000))
                ticks = mt5.copy_ticks_range(sym, server_dt(last / 1000), server_dt(now_srv + 60),
                                             mt5.COPY_TICKS_ALL)
                if ticks is None:
                    logger.warning(f"{sym}: copy_ticks_range failed {mt5.last_error()}")
                    continue
                ticks = ticks[ticks["time_msc"] > last]
                if len(ticks):
                    write_ticks(sym, ticks)
                    state[sym] = int(ticks["time_msc"][-1])
                    totals[sym] += len(ticks)
            STATE.write_text(json.dumps(state))
            if time.time() - last_report > 3600:
                logger.info(f"ticks written last hour: {totals}")
                totals = {s: 0 for s in symbols}
                last_report = time.time()
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            logger.info("stopped by user")
            break
        except Exception as e:
            logger.exception(f"loop error: {e}")
            time.sleep(60)
            if mt5.terminal_info() is None:
                MT5Client().connect()


if __name__ == "__main__":
    main()
