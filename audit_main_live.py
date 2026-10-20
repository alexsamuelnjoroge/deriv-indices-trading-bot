"""
Live audit of the main bot's strategies from Deriv's profit_table (real fills,
all costs included). Contracts are mapped to config entries by symbol +
contract type + duration / multiplier parsed from the shortcode.

Usage:
    python audit_main_live.py [--since 2026-08-01] [--split 2026-10-01]
"""

import argparse
import asyncio
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient

logger.remove()


def label(c: dict) -> str:
    sc = c.get("shortcode", "")
    sym = c.get("underlying_symbol") or c.get("underlying") or ""
    ct = sc.split("_")[0]
    if ct in ("CALL", "PUT", "CALLE", "PUTE"):
        dur = int(c["sell_time"]) - int(c["purchase_time"])
        m = re.search(r"_(\d+)T_", sc)
        d = f"{m.group(1)}t" if m else f"{round(dur / 60)}m"
        return f"{sym} Rise/Fall {d}"
    if ct in ("MULTUP", "MULTDOWN"):
        parts = sc.replace(f"_{sym.upper()}_", "_SYM_", 1).split("_")   # symbols like R_50 contain "_"
        mult = parts[3] if len(parts) > 3 else "?"
        return f"{sym} multiplier x{mult}"
    if ct == "ACCU":
        return f"{sym} accumulator"
    return f"{sym} {ct}"


async def fetch(since: datetime) -> list[dict]:
    load_dotenv(".env")
    c = DerivClient(os.getenv("DERIV_API_TOKEN"))
    await c.connect()
    rows, offset, cutoff = [], 0, since.timestamp()
    while True:                                   # newest first, pages of 100 (500 errors)
        for wait in (2, 10, 30, 60, 120):
            try:
                r = await c._send({"profit_table": 1, "description": 1, "limit": 100, "offset": offset})
                break
            except RuntimeError as e:
                if "RateLimit" not in str(e):
                    raise
                await asyncio.sleep(wait)
        else:
            raise RuntimeError("profit_table rate limit persisted")
        await asyncio.sleep(1.0)
        if offset % 2000 == 0:
            print(f"  fetched {offset} ...", flush=True)
        batch = r["profit_table"]["transactions"]
        rows += [t for t in batch if int(t["purchase_time"]) >= cutoff]
        if len(batch) < 100 or int(batch[-1]["purchase_time"]) < cutoff:
            break
        offset += 100
    await c.disconnect()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-01")
    ap.add_argument("--split", default="2026-10-01")
    a = ap.parse_args()
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc)
    split = datetime.fromisoformat(a.split).replace(tzinfo=timezone.utc).timestamp()
    cache = Path(f"data/profit_table_{a.since}.json")
    if cache.exists() and time.time() - cache.stat().st_mtime < 6 * 3600:
        rows = json.loads(cache.read_text())
    else:
        rows = asyncio.run(fetch(since))
        cache.write_text(json.dumps(rows))
    print(f"{len(rows)} settled contracts since {a.since}")
    if rows and "shortcode" not in rows[0]:
        print("sample keys:", list(rows[0].keys()))

    g = defaultdict(list)
    for t in rows:
        buy, sell = float(t["buy_price"]), float(t["sell_price"])
        g[label(t)].append((int(t["purchase_time"]), buy, sell - buy))

    print(f"\n{'group':<34}{'period':<10}{'n':>6}{'WR%':>7}{'staked':>9}{'net $':>9}{'ret/stake':>10}{'t':>6}")
    for k in sorted(g, key=lambda k: sum(x[2] for x in g[k])):
        if len(g[k]) < 10:
            continue
        for name, sel in (("all", lambda x: True), ("pre-Oct", lambda x: x[0] < split), ("Oct+", lambda x: x[0] >= split)):
            xs = [x for x in g[k] if sel(x)]
            if len(xs) < (10 if name == 'all' else 2):
                continue
            rets = [p / b for _, b, p in xs]
            mu = sum(rets) / len(rets)
            sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) or 1e-9
            print(f"{k if name == 'all' else '':<34}{name:<10}{len(xs):>6}"
                  f"{sum(p > 0 for _, _, p in xs) / len(xs) * 100:>7.1f}{sum(b for _, b, _ in xs):>9.2f}"
                  f"{sum(p for *_, p in xs):>9.2f}{mu * 100:>9.1f}%{mu / sd * math.sqrt(len(xs)):>6.1f}")
    tot = [x for v in g.values() for x in v]
    print(f"\nTOTAL net ${sum(p for *_, p in tot):+.2f} on ${sum(b for _, b, _ in tot):.2f} staked")


if __name__ == "__main__":
    main()
