"""Sequential, rate-limit-aware Dukascopy M5 bid fetcher for the FX month-end-fix study.
Caches data/duka/frx<PAIR>_M5_<YEAR>.csv (same format as the existing caches).
Validates each year (bars per weekday-month) and refetches files that look truncated.
Usage: python research/_fetch_duka_fx.py EURUSD GBPUSD ...
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import dukascopy_python as d
from dukascopy_python import instruments as ins, _fetch

CACHE = Path(__file__).resolve().parent.parent / "data" / "duka"
CACHE.mkdir(parents=True, exist_ok=True)
END = datetime(2026, 10, 20, tzinfo=timezone.utc)


def fetch_range(inst, a, b):
    rows, cur, first = [], int(a.timestamp() * 1000), True
    endms = b.timestamp() * 1000
    fails = 0
    while True:
        try:
            r = _fetch(inst, d.INTERVAL_MIN_5, d.OFFER_SIDE_BID, cur, limit=30000)
        except Exception as e:  # network / json
            fails += 1
            print("   err", e, flush=True)
            time.sleep(min(60, 5 * fails))
            continue
        if isinstance(r, dict):
            fails += 1
            print("   api:", str(r)[:80], flush=True)
            time.sleep(min(120, 15 * fails))
            if fails > 40:
                raise RuntimeError("too many failures")
            continue
        fails = 0
        if not first and r and r[0][0] == cur:
            r = r[1:]
        first = False
        if not r:
            break
        done = False
        for row in r:
            if row[0] >= endms:
                done = True
                break
            rows.append(row[:6])
            cur = row[0]
        if done:
            break
        time.sleep(1.0)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def ok(f, y):
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
    except Exception:
        return False
    if len(df) == 0:
        return False
    idx = df.index
    wk = idx[idx.dayofweek < 5]
    last_m = 12 if y < 2026 else 9
    months = pd.Series(1, index=wk).groupby(wk.month).size()
    return all(months.get(m, 0) > 4500 for m in range(1, last_m + 1))


for pair in sys.argv[1:]:
    inst = getattr(ins, f"INSTRUMENT_FX_MAJORS_{pair[:3]}_{pair[3:]}")
    for y in range(2010, 2027):
        f = CACHE / f"frx{pair}_M5_{y}.csv"
        if f.exists() and ok(f, y):
            continue
        a = datetime(y, 1, 1, tzinfo=timezone.utc)
        b = min(datetime(y + 1, 1, 1, tzinfo=timezone.utc), END)
        probe = _fetch(inst, d.INTERVAL_MIN_5, d.OFFER_SIDE_BID, int(a.timestamp() * 1000), limit=10)
        if isinstance(probe, list) and probe and probe[0][0] >= b.timestamp() * 1000:
            print(f"{pair} {y}: no M5 history on feed (first bar {datetime.fromtimestamp(probe[0][0]/1000, timezone.utc):%Y-%m-%d}), skip", flush=True)
            continue
        parts = []
        m = a
        while m < b:
            n = datetime(m.year + (m.month == 12), m.month % 12 + 1, 1, tzinfo=timezone.utc)
            parts.append(fetch_range(inst, m, min(n, b)))
            m = n
        df = pd.concat(parts)
        df = df[~df.index.duplicated()].sort_index()
        df.to_csv(f)
        print(f"{pair} {y}: {len(df)} bars, ok={ok(f, y)}", flush=True)
