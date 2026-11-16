"""
Session-range breakout across markets (pre-registered 2026-11-13).

Same mechanics as the gold rule (incubator D): range = high/low of a quiet window, OCO stop
orders at the range, SL = opposite side, TP = 1R, orders cancelled at `cancel`, trade closed
at `flat` (server time = New York + 7h). Ambiguity booked against us. One window per market,
fixed before running (no per-market tuning):
  metals/FX : 03:00-08:00 range (Asian session), cancel 22:00, flat 23:00 — identical to gold
  DE40      : 03:00-10:00 range (until the 09:00 CET Frankfurt open), cancel 17:00, flat 18:30
  US500/TEC : 01:00-16:30 range (Globex reopen to the 09:30 ET cash open), cancel 22:00, flat 22:55
Costs: IC Markets MT5 round trip measured 2026-11-13 (median spread 08-23h + commission).
Split: FX 2010-16 | 2017-26; silver/indices (data from 2018) 2018-21 | 2022-26.
Pass: TRAIN mean > 0, t >= 2.5; TEST mean > 0, t >= 1.5 and > 0 at 2x cost (~10 markets).
Passers are then re-run on IC MT5 M15 data (research/xasset_breakout_battery.py --mt5).

    python research/xasset_breakout_battery.py [--mt5 SYMBOL ...]
"""

import glob
import math
import sys

import numpy as np
import pandas as pd

ASIA = dict(r0=3 * 60, r1=8 * 60, cancel=22 * 60, flat=23 * 60)
MARKETS = {
    # name: (Dukascopy file prefix, MT5 symbol, session, test-start year, round-trip cost bp)
    "XAUUSD": ("frxXAUUSD", "XAUUSD", ASIA, 2017, 0.36),
    "XAGUSD": ("XAGUSD", "XAGUSD", ASIA, 2022, None),
    "EURUSD": ("frxEURUSD", "EURUSD", ASIA, 2017, None),
    "GBPUSD": ("frxGBPUSD", "GBPUSD", ASIA, 2017, None),
    "AUDUSD": ("frxAUDUSD", "AUDUSD", ASIA, 2017, None),
    "USDJPY": ("frxUSDJPY", "USDJPY", ASIA, 2017, None),
    "USDCHF": ("frxUSDCHF", "USDCHF", ASIA, 2017, None),
    "USDCAD": ("frxUSDCAD", "USDCAD", ASIA, 2017, None),
    "DE40": ("DEU40", "DE40", dict(r0=3 * 60, r1=10 * 60, cancel=17 * 60, flat=18 * 60 + 30), 2022, None),
    "US500": ("USA500", "US500", dict(r0=60, r1=16 * 60 + 30, cancel=22 * 60, flat=22 * 60 + 55), 2022, None),
    "USTEC": ("USATECH", "USTEC", dict(r0=60, r1=16 * 60 + 30, cancel=22 * 60, flat=22 * 60 + 55), 2022, None),
}
COSTS = {}   # filled from MARKETS / measured values below


def load_duka_m15(prefix):
    parts = [p for p in (pd.read_csv(f, index_col=0, parse_dates=True)
                         for f in sorted(glob.glob(f"data/duka/{prefix}_M5_*.csv"))) if len(p)]
    m5 = pd.concat(parts)
    m5 = m5[~m5.index.duplicated()].sort_index()
    if "volume" in m5:
        m5 = m5[m5["volume"] > 0]
    ny = m5.index.tz_convert("America/New_York")
    m5.index = (ny + pd.Timedelta(hours=7)).tz_localize(None)
    m15 = m5.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    m15["mins"] = m15.index.hour * 60 + m15.index.minute
    return m15


def load_mt5_m15(symbol, count=99_000):
    import MetaTrader5 as mt5
    mt5.initialize()
    mt5.symbol_select(symbol, True)
    r = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, count)
    mt5.shutdown()
    d = pd.DataFrame(r)
    f = pd.DataFrame({"open": d.open.values, "high": d.high.values, "low": d.low.values,
                      "close": d.close.values}, index=pd.to_datetime(d.time, unit="s"))
    f["mins"] = f.index.hour * 60 + f.index.minute
    return f


def run(m15, sess, bar_min=15, skip=frozenset(), early_flat=None):
    """Returns list of (date, side, gross bp, risk bp). bar_min = bar size of the input frame.
    skip: server dates with no trading. early_flat: {server date: minute} -> no fills from that
    minute and any open trade is closed at that minute's open."""
    need = (sess["r1"] - sess["r0"]) // bar_min
    early_flat = early_flat or {}
    out = []
    for d, g in m15.groupby(m15.index.date):
        if g.index[0].weekday() >= 5 or d in skip:
            continue
        s = sess
        if d in early_flat:
            s = {**sess, "cancel": min(sess["cancel"], early_flat[d]), "flat": min(sess["flat"], early_flat[d])}
        rng = g[(g["mins"] >= s["r0"]) & (g["mins"] < s["r1"])]
        if len(rng) < 0.9 * need:
            continue
        hi, lo = rng["high"].max(), rng["low"].min()
        if hi <= lo:
            continue
        after = g[g["mins"] >= s["r1"]].reset_index(drop=True)
        for k in range(len(after)):
            b = after.iloc[k]
            if b["mins"] >= s["cancel"]:
                break
            up, dn = b["high"] > hi, b["low"] < lo
            if not (up or dn):
                continue
            if up and dn:
                side = 1 if (hi - b["open"]) <= (b["open"] - lo) else -1
                entry = hi if side > 0 else lo
                stop = lo if side > 0 else hi
                out.append((d, side, side * (stop / entry - 1) * 1e4, abs(entry - stop) / entry * 1e4))
                break
            side = 1 if up else -1
            entry = max(b["open"], hi) if up else min(b["open"], lo)
            stop = lo if up else hi
            target = entry + side * abs(entry - stop)
            px = stop if ((side > 0 and b["low"] <= stop) or (side < 0 and b["high"] >= stop)) else None
            for j in range(k + 1, len(after)):
                if px is not None:
                    break
                c = after.iloc[j]
                if c["mins"] >= s["flat"]:
                    px = c["open"]
                elif (side > 0 and c["low"] <= stop) or (side < 0 and c["high"] >= stop):
                    px = stop
                elif (side > 0 and c["high"] >= target) or (side < 0 and c["low"] <= target):
                    px = target
            if px is None:
                px = after.iloc[-1]["close"]
            out.append((d, side, side * (px / entry - 1) * 1e4, abs(entry - stop) / entry * 1e4))
            break
    return out


def stats(tr, cost):
    if len(tr) < 30:
        return None
    net = np.array([x[2] for x in tr]) - cost
    r = net / np.array([x[3] for x in tr])
    sd = net.std(ddof=1)
    return dict(n=len(net), tpy=len(net) / len({x[0].year for x in tr}), mean=net.mean(),
                t=net.mean() / sd * math.sqrt(len(net)) if sd > 0 else 0, R=r.mean(),
                wr=(net > 0).mean() * 100)


def fmt(s):
    return "n<30" if not s else (f"n={s['n']:4d} /yr={s['tpy']:4.0f} {s['mean']:+6.2f}bp "
                                 f"{s['R']:+.3f}R t={s['t']:+5.2f} WR={s['wr']:4.1f}%")


def evaluate(name, tr, test_year, cost):
    a = [x for x in tr if x[0].year < test_year]
    b = [x for x in tr if x[0].year >= test_year]
    sa, sb, sb2 = stats(a, cost), stats(b, cost), stats(b, 2 * cost)
    ok = bool(sa and sb and sb2 and sa["mean"] > 0 and sa["t"] >= 2.5 and sb["mean"] > 0
              and sb["t"] >= 1.5 and sb2["mean"] > 0)
    print(f"\n[{'PASS' if ok else 'fail'}] {name} (cost {cost:.2f}bp RT, test from {test_year})")
    print(f"   TRAIN {fmt(sa)}")
    print(f"   TEST  {fmt(sb)}")
    for side, lab in ((1, "long "), (-1, "short")):
        print(f"     {lab} train {fmt(stats([x for x in a if x[1] == side], cost))} | "
              f"test {fmt(stats([x for x in b if x[1] == side], cost))}")
    yr = {}
    for x in tr:
        yr.setdefault(x[0].year, []).append((x[2] - cost) / x[3])
    print("   years (R): " + " ".join(f"{y % 100:02d}:{np.mean(v):+.2f}" for y, v in sorted(yr.items())), flush=True)
    return ok


if __name__ == "__main__":
    import json
    measured = json.load(open("research/xasset_costs.json"))
    if "--mt5" in sys.argv:
        for name in sys.argv[sys.argv.index("--mt5") + 1:]:
            prefix, sym, sess, _, cost = MARKETS[name]
            cost = cost if cost is not None else measured[name]
            m = load_mt5_m15(sym)
            tr = run(m, sess)
            s = stats(tr, cost)
            print(f"\nIC MT5 {name} {m.index[0]:%Y-%m-%d} -> {m.index[-1]:%Y-%m-%d}: {fmt(s)}")
            yr = {}
            for x in tr:
                yr.setdefault(x[0].year, []).append((x[2] - cost) / x[3])
            print("   years (R): " + " ".join(f"{y}:{np.mean(v):+.2f}" for y, v in sorted(yr.items())))
    else:
        for name, (prefix, sym, sess, test_year, cost) in MARKETS.items():
            cost = cost if cost is not None else measured[name]
            evaluate(name, run(load_duka_m15(prefix), sess), test_year, cost)
