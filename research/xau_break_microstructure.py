"""Helper for xau_break_drift_battery: is the break drift a bid/spread artifact?

Profiles returns segment by segment around the 00:00-01:00 server halt for
Dukascopy (bid), Deriv (its own quote) and MT5 ticks (bid / ask / mid), and
recomputes the trade with real ask-in / bid-out fills. Read-only MT5 access.
"""
import json
import math

import numpy as np
import pandas as pd
import MetaTrader5 as mt5


def t(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * math.sqrt(len(x)) if len(x) > 2 else np.nan


def s(x):
    x = pd.Series(x).dropna()
    return f"n={len(x):4d} mean={x.mean():+6.2f}bp t={t(x):+5.2f} hit={np.mean(x>0)*100:4.1f}%"


def to_srv(ts_utc):
    return ts_utc.dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=7)


# ---------------- Deriv 5-min profile
dj = pd.DataFrame(json.load(open("data/pro/deriv_XAUUSD_300.json"))).astype(float)
dj["srv"] = to_srv(pd.to_datetime(dj.epoch, unit="s", utc=True))
dj = dj.drop_duplicates("srv").set_index("srv")
dclose = pd.Series(dj.close.values, index=dj.index + pd.Timedelta(minutes=5))
dopen = dj.open
days = pd.DatetimeIndex(sorted(set(dj.index.normalize())))
days = days[days.weekday.isin([1, 2, 3, 4])][1:]
print("=== Deriv 5-min quotes (last ~1y) segments around the halt (Tue-Fri) ===")
p = lambda ser, tt: ser.reindex(tt).values
D = days
segs = {
    "23:55->halt": (p(dclose, D) / p(dclose, D - pd.Timedelta(minutes=5)) - 1) * 1e4,
    "halt->reopen open": (p(dopen, D + pd.Timedelta(minutes=60)) / p(dclose, D) - 1) * 1e4,
    "reopen open->01:05": (p(dclose, D + pd.Timedelta(minutes=65)) / p(dopen, D + pd.Timedelta(minutes=60)) - 1) * 1e4,
    "01:05->03:00": (p(dclose, D + pd.Timedelta(minutes=180)) / p(dclose, D + pd.Timedelta(minutes=65)) - 1) * 1e4,
    "23:55->03:00 (trade)": (p(dclose, D + pd.Timedelta(minutes=180)) / p(dclose, D - pd.Timedelta(minutes=5)) - 1) * 1e4,
    "01:05->03:00 only (buy after reopen)": (p(dclose, D + pd.Timedelta(minutes=180)) / p(dclose, D + pd.Timedelta(minutes=65)) - 1) * 1e4,
}
for k, v in segs.items():
    print(f"  {k:38s} {s(v)}")

# ---------------- MT5 ticks
mt5.initialize()
rows = []
prof = []
alld = pd.date_range("2026-03-25", "2026-10-21", freq="D")
alld = alld[alld.weekday.isin([1, 2, 3, 4])]
for D0 in alld:
    tk = mt5.copy_ticks_range("XAUUSD", (D0 - pd.Timedelta(hours=1)).to_pydatetime(),
                              (D0 + pd.Timedelta(hours=8)).to_pydatetime(), mt5.COPY_TICKS_ALL)
    if tk is None or len(tk) == 0:
        continue
    tk = pd.DataFrame(tk)
    tk["s"] = pd.to_datetime(tk.time_msc, unit="ms")
    tk = tk.set_index("s").sort_index()
    tk["mid"] = (tk.bid + tk.ask) / 2

    def at(tt):
        x = tk.loc[:tt]
        return x.iloc[-1] if len(x) else None

    if tk.index[-1] < D0 + pd.Timedelta(minutes=175) or tk.index[0] > D0 - pd.Timedelta(minutes=10):
        continue
    post = tk.loc[D0 + pd.Timedelta(minutes=55):]
    if len(post) == 0:
        continue
    first = post.iloc[0]
    q = {k: at(D0 + pd.Timedelta(minutes=m)) for k, m in
         [("e55", -5), ("e56", -4), ("e57", -3), ("e59", -1), ("x65", 65), ("x90", 90), ("x180", 180), ("x195", 195)]}
    q["halt"] = at(D0)
    if any(v is None for v in q.values()):
        continue
    bp = lambda a, b: (b / a - 1) * 1e4
    rows.append(dict(
        D=D0, first_tick=post.index[0],
        spr55=bp(q["e55"].bid, q["e55"].ask), spr_halt=bp(q["halt"].bid, q["halt"].ask),
        spr_open=bp(first.bid, first.ask), spr65=bp(q["x65"].bid, q["x65"].ask), spr180=bp(q["x180"].bid, q["x180"].ask),
        bid_55_halt=bp(q["e55"].bid, q["halt"].bid), mid_55_halt=bp(q["e55"].mid, q["halt"].mid), ask_55_halt=bp(q["e55"].ask, q["halt"].ask),
        mid_halt_open=bp(q["halt"].mid, first.mid), mid_open_65=bp(first.mid, q["x65"].mid), bid_open_65=bp(first.bid, q["x65"].bid),
        mid_65_180=bp(q["x65"].mid, q["x180"].mid),
        mid_trade=bp(q["e55"].mid, q["x180"].mid), bid_trade=bp(q["e55"].bid, q["x180"].bid),
        real_trade=bp(q["e55"].ask, q["x180"].bid),
        real_e56=bp(q["e56"].ask, q["x180"].bid), real_e57=bp(q["e57"].ask, q["x180"].bid), real_e59=bp(q["e59"].ask, q["x180"].bid),
        real_halt=bp(q["halt"].ask, q["x180"].bid), real_x195=bp(q["e55"].ask, q["x195"].bid),
        real_after_reopen=bp(q["x65"].ask, q["x180"].bid),
    ))
mt5.shutdown()
R = pd.DataFrame(rows).set_index("D")
print(f"\n=== MT5 IC Markets ticks, {R.index[0].date()}..{R.index[-1].date()}, n={len(R)} Tue-Fri breaks ===")
print("median spreads (bp): " + ", ".join(f"{c}={R[c].median():.2f}" for c in ["spr55", "spr_halt", "spr_open", "spr65", "spr180"]))
print("mean spreads (bp):   " + ", ".join(f"{c}={R[c].mean():.2f}" for c in ["spr55", "spr_halt", "spr_open", "spr65", "spr180"]))
for c in ["bid_55_halt", "mid_55_halt", "ask_55_halt", "mid_halt_open", "mid_open_65", "bid_open_65", "mid_65_180",
          "mid_trade", "bid_trade", "real_trade", "real_e56", "real_e57", "real_e59", "real_halt", "real_x195", "real_after_reopen"]:
    print(f"  {c:20s} {s(R[c])}")
print(f"  real_trade net Deriv 1.2bp: {s(R.real_trade-1.2)} ; net MT5 swap: {s(R.real_trade - np.where(R.index.weekday==3, 4.02, 1.34))}")
print(f"  real_trade without top 3 days: {s(R.real_trade.sort_values().iloc[:-3])}; median {R.real_trade.median():+.2f}")
# Same window on Dukascopy for the same dates (bid)
