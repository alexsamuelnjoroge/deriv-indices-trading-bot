"""
XAUUSD intraday event-time research (hypotheses 3-5).

Data: Dukascopy 5-min bid candles 2018 -> now (data/duka/frxXAUUSD_M5_*.csv, UTC).
Split: DEV 2018-2021 | VAL 2022 -> 2024-07 | RECENT 2024-08 -> now.
A candidate needs the same sign and |t| >= 2.5 in BOTH DEV and VAL (many windows
are tested, so 2.0 would let noise through). Returns in basis points, gross;
round-trip cost is ~0.5-1.2bp depending on venue.

Windows (local times converted to UTC with DST):
  * London AM fix 10:30 and PM fix 15:00 (Europe/London): 30 min before, the
    15 min of the auction, 60 min after
  * US data 08:30 New York (13:30/12:30 UTC): 30 min before, first 5 min,
    5-60 min after (drift) and that drift signed by the first-5-min direction
  * NFP (first Friday of the month) separately
  * Session opens: Tokyo 09:00, London 08:00, New York 09:30 local — first hour,
    and next hour signed by the first hour (momentum vs reversal)
  * Month-end: last trading day 15:00 London -> close
"""

import glob
import math
from datetime import time

import numpy as np
import pandas as pd

PER = {"DEV": ("2018-01-01", "2022-01-01"), "VAL": ("2022-01-01", "2024-08-01"),
       "RECENT": ("2024-08-01", "2100-01-01")}


def load() -> pd.Series:
    parts = [pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(glob.glob("data/duka/frxXAUUSD_M5_*.csv"))]
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    return df["close"]


def window_returns(close: pd.Series, tz: str, start: time, end: time, days=None) -> pd.Series:
    """bp return from the last close at/before local `start` to the last close at/before local `end`, per day."""
    loc = close.tz_convert(tz)
    out = {}
    for day, g in loc.groupby(loc.index.date):
        if days is not None and pd.Timestamp(day) not in days:
            continue
        t0 = g[g.index.time <= start]
        t1 = g[g.index.time <= end]
        if t0.empty or t1.empty or t1.index[-1] <= t0.index[-1]:
            continue
        if (t1.index[-1] - t0.index[-1]) > pd.Timedelta(hours=8):
            continue
        out[pd.Timestamp(day)] = (t1.iloc[-1] / t0.iloc[-1] - 1) * 1e4
    return pd.Series(out, dtype=float)


def stat(x: pd.Series):
    x = x.dropna()
    if len(x) < 20:
        return None
    return x.mean(), len(x), x.mean() / x.std() * math.sqrt(len(x))


def report(label: str, s: pd.Series):
    cells, st = [], {}
    for p, (a, b) in PER.items():
        r = stat(s[(s.index >= a) & (s.index < b)])
        st[p] = r
        cells.append(f"{p} {r[0]:+6.2f}bp t={r[2]:+4.1f} n={r[1]:<4}" if r else f"{p} {'--':<24}")
    ok = (st["DEV"] and st["VAL"] and abs(st["DEV"][2]) >= 2.5 and abs(st["VAL"][2]) >= 2.5
          and np.sign(st["DEV"][0]) == np.sign(st["VAL"][0]))
    print(f"  {label:<46} " + "  ".join(cells) + ("  <<<" if ok else ""))


def first_fridays(idx) -> set:
    days = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in idx}))
    fr = days[days.dayofweek == 4]
    return {d for d in fr if d.day <= 7}


def main():
    c = load()
    print(f"XAUUSD M5 {c.index[0].date()} -> {c.index[-1].date()} ({len(c):,} bars)\n")
    T = lambda h, m=0: time(h, m)

    print("== London fixes (Europe/London local time)")
    for name, fix in (("AM fix 10:30", T(10, 30)), ("PM fix 15:00", T(15, 0))):
        h, m = fix.hour, fix.minute
        report(f"{name}: 30m before", window_returns(c, "Europe/London", T(h if m else h - 1, (m - 30) % 60), fix))
        report(f"{name}: auction 15m", window_returns(c, "Europe/London", fix, T(h, m + 15)))
        report(f"{name}: 15-75m after", window_returns(c, "Europe/London", T(h, m + 15), T(h + 1, m + 15)))

    print("\n== US data time 08:30 New York")
    pre = window_returns(c, "America/New_York", T(8, 0), T(8, 30))
    first = window_returns(c, "America/New_York", T(8, 30), T(8, 35))
    drift = window_returns(c, "America/New_York", T(8, 35), T(9, 30))
    report("30m before 08:30", pre)
    report("first 5m after 08:30", first)
    report("08:35-09:30 drift", drift)
    report("08:35-09:30 x sign(first 5m) (+=follow)", drift * np.sign(first.reindex(drift.index)))
    nfp = first_fridays(c.tz_convert("America/New_York").index.date)
    nf_first = window_returns(c, "America/New_York", T(8, 30), T(8, 35), nfp)
    nf_drift = window_returns(c, "America/New_York", T(8, 35), T(10, 30), nfp)
    report("NFP Fridays: first 5m", nf_first)
    report("NFP: 08:35-10:30 x sign(first 5m)", nf_drift * np.sign(nf_first.reindex(nf_drift.index)))

    print("\n== Session opens: first hour, then next hour signed by first hour (+ = momentum)")
    for name, tz, (h, m) in (("Tokyo 09:00", "Asia/Tokyo", (9, 0)), ("London 08:00", "Europe/London", (8, 0)),
                             ("New York 09:30", "America/New_York", (9, 30))):
        h1 = window_returns(c, tz, T(h, m), T(h + 1, m))
        h2 = window_returns(c, tz, T(h + 1, m), T(h + 2, m))
        report(f"{name}: first hour", h1)
        report(f"{name}: 2nd hour x sign(1st)", h2 * np.sign(h1.reindex(h2.index)))

    print("\n== Month-end: last trading day, 15:00 London -> 16:00 New York")
    loc = c.tz_convert("Europe/London")
    days = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in loc.index.date}))
    last_days = set(days.to_series().groupby(days.to_period("M")).max())
    me = window_returns(c, "Europe/London", T(15, 0), T(21, 0), last_days)
    other = window_returns(c, "Europe/London", T(15, 0), T(21, 0))
    report("month-end 15:00-21:00 London", me)
    report("all days 15:00-21:00 London (baseline)", other)
    print("\n('<<<' = same sign and |t| >= 2.5 in DEV and VAL; costs ~0.5-1.2bp per round trip)")


if __name__ == "__main__":
    main()
