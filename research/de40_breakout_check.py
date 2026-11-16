"""
DE40 session-range breakout — independent confirmation (pre-registered 2026-11-13).

DE40 was flagged from 2018-2026 data (xasset_breakout_battery.py: +0.03R / +0.06R, t < 1.6),
so those years cannot confirm it. Dukascopy DAX data before 2018 only covers Frankfurt cash
hours (no 03:00-10:00 pre-open bars), so the clean test uses IC Markets MT5 DE40 H1 history,
which has pre-open bars from 2016: 2016-2017 were never looked at.
Rule unchanged: range 03:00-10:00 server, OCO stops, SL opposite side, TP 1R, cancel 17:00,
flat 18:30. H1 bars book more same-bar stop/target ambiguities as losses (conservative).
Cost 0.19bp RT (IC MT5 measured). Pass: 2016-2017 mean > 0 with t >= 2, and > 0 at 2x cost.
Result 2026-11-13: -0.088R per trade (n=481), both years negative -> DE40 dropped.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xasset_breakout_battery import MARKETS, fmt, run, stats  # noqa: E402

COST = 0.19
H1_FILE = Path("data/pro/mt5_DE40_H1_full.csv")


def load_h1():
    if not H1_FILE.exists():
        import MetaTrader5 as mt5
        mt5.initialize()
        mt5.symbol_select("DE40", True)
        r = mt5.copy_rates_from_pos("DE40", mt5.TIMEFRAME_H1, 1, 99000)
        mt5.shutdown()
        d = pd.DataFrame(r)
        d.index = pd.to_datetime(d.time, unit="s")
        d.to_csv(H1_FILE)
    d = pd.read_csv(H1_FILE, index_col=0, parse_dates=True)
    f = d[["open", "high", "low", "close"]].copy()
    f["mins"] = f.index.hour * 60 + f.index.minute
    return f


if __name__ == "__main__":
    tr = run(load_h1(), MARKETS["DE40"][2], bar_min=60)
    new = [x for x in tr if x[0].year in (2016, 2017)]
    s, s2 = stats(new, COST), stats(new, 2 * COST)
    ok = bool(s and s2 and s["mean"] > 0 and s["t"] >= 2 and s2["mean"] > 0)
    print(f"[{'PASS' if ok else 'fail'}] DE40 2016-2017 (never seen, IC MT5 H1): {fmt(s)}")
    print(f"   at 2x cost: {fmt(s2)}")
    print(f"   2018-2026 same H1 data (spotting period): {fmt(stats([x for x in tr if x[0].year >= 2018], COST))}")
    yr = {}
    for x in tr:
        yr.setdefault(x[0].year, []).append((x[2] - COST) / x[3])
    print("   years (R): " + " ".join(f"{y % 100:02d}:{np.mean(v):+.2f}" for y, v in sorted(yr.items())))
