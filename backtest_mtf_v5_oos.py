"""
Out-of-sample test of the main bot's mtf_v5 multiplier strategy
(src/strategies/mtf_v5.py) for the three configs in config.yaml.

mtf_v5 was tuned on ~1 year of Deriv data ending 2026-08-03, so:
  OOS-A 2018-01 -> 2025-07 (never seen) | IS 2025-08 -> 2026-08-03 | OOS-B 2026-08-04 -> now

Data: Dukascopy 5-min bid candles (dukascopy-python), cached to data/duka/.
H1/D1 closes are resampled from them. The live strategy class is driven bar by
bar (_evaluate_bar), with the config's session_peak plus the global
blocked_hours_eat from config.yaml applied.

Exits follow Deriv multiplier rules: SL/TP are dollar amounts that include the
commission charged at open, so the price must move (sl_pct - comm) against or
(tp_pct + comm) in favour. Win = +tp_rr R, loss = -1R. One contract at a time
(max_open_contracts: 1). Same-bar SL+TP counts as a loss. Open after 3 days ->
marked to market.

Usage:
    python backtest_mtf_v5_oos.py
"""

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.strategies.mtf_v5 import MTFV5Strategy

CACHE = Path("data/duka")
START, END = datetime(2018, 1, 1), datetime(2026, 10, 16)
COMM = {"frxXAUUSD": 0.00012, "frxEURUSD": 0.00024, "frxUSDJPY": 0.00028}
MAX_HOLD = 3 * 288
PERIODS = {"OOS-A 2018-25": ("2018-01-01", "2025-08-01"),
           "IS 2025-26": ("2025-08-01", "2026-08-04"),
           "OOS-B Aug26+": ("2026-08-04", "2100-01-01")}


def fetch(sym: str) -> pd.DataFrame:
    import dukascopy_python as d
    from dukascopy_python import instruments as ins
    inst = {"frxXAUUSD": ins.INSTRUMENT_FX_METALS_XAU_USD,
            "frxEURUSD": ins.INSTRUMENT_FX_MAJORS_EUR_USD,
            "frxUSDJPY": ins.INSTRUMENT_FX_MAJORS_USD_JPY}[sym]
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in range(START.year, END.year + 1):
        f = CACHE / f"{sym}_M5_{y}.csv"
        if not f.exists():
            a, b = datetime(y, 1, 1), min(datetime(y + 1, 1, 1), END)
            df = d.fetch(inst, d.INTERVAL_MIN_5, d.OFFER_SIDE_BID, a, b)
            df.to_csv(f)
            print(f"    fetched {sym} {y}: {len(df)} bars", flush=True)
        parts.append(pd.read_csv(f, index_col=0, parse_dates=True))
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close"]]


def run(sym: str, cfg: dict, blocked_eat: set) -> list[tuple[int, float]]:
    m5 = fetch(sym)
    h1 = m5["close"].resample("1h", label="right", closed="left").last().dropna()
    d1 = m5["close"].resample("1D", label="right", closed="left").last().dropna()
    secs = lambda idx: ((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
    ep5 = secs(m5.index) + 300                                          # bar close epoch
    eph, epd = secs(h1.index), secs(d1.index)
    o, hi, lo, c = (m5[k].to_numpy() for k in ("open", "high", "low", "close"))
    hv, dv = h1.to_numpy(), d1.to_numpy()

    st = MTFV5Strategy(dict(cfg))
    comm = COMM[sym]
    ih = idd = 0
    trades, busy_until = [], -1
    for i in range(len(c)):
        t = ep5[i]
        while ih < len(eph) and eph[ih] <= t:
            st._htf_closes.append(hv[ih]); ih += 1
        while idd < len(epd) and epd[idd] <= t:
            st._daily_closes.append(dv[idd]); idd += 1
        st._ltf_bars.append({"epoch": t, "open": o[i], "high": hi[i], "low": lo[i], "close": c[i]})
        if len(st._ltf_bars) > 400:
            del st._ltf_bars[:-200]
        if len(st._htf_closes) > 1200:
            del st._htf_closes[:-800]
        if len(st._daily_closes) > 400:
            del st._daily_closes[:-250]
        if i <= busy_until:
            continue
        if ((datetime.fromtimestamp(t, timezone.utc).hour + 3) % 24) in blocked_eat:
            continue
        sig = st._evaluate_bar(c[i], t)
        if sig.action not in ("BUY_RISE", "BUY_FALL"):
            continue
        d = 1 if sig.action == "BUY_RISE" else -1
        entry = c[i]
        stop = entry * (1 - d * (sig.sl_pct - comm))
        tp = entry * (1 + d * (sig.tp_pct + comm))
        rr = sig.tp_pct / sig.sl_pct
        r, j_end = None, min(i + MAX_HOLD, len(c) - 1)
        for j in range(i + 1, j_end + 1):
            if (d == 1 and lo[j] <= stop) or (d == -1 and hi[j] >= stop):
                r = -1.0
            elif (d == 1 and hi[j] >= tp) or (d == -1 and lo[j] <= tp):
                r = rr
            if r is not None:
                break
        if r is None:
            j = j_end
            r = (d * (c[j] / entry - 1) - comm) / sig.sl_pct
        trades.append((int(t), r))
        busy_until = j
    return trades


def stat(rs):
    if len(rs) < 2:
        return None
    a = np.array(rs)
    return dict(n=len(a), wr=(a > 0).mean(), ev=a.mean(), net=a.sum(), t=a.mean() / a.std(ddof=1) * math.sqrt(len(a)))


def main():
    cfg_all = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    blocked = set(cfg_all["risk"].get("blocked_hours_eat", []))
    entries = [e for e in cfg_all["symbols"] if e.get("strategy_type") == "mtf_v5"]
    ts = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())
    for e in entries:
        sym = e["symbol"]
        cfg = {**cfg_all.get("strategy", {}), **e}
        print(f"\n== {sym}  ema{cfg['ema_period']} rsi{cfg['rsi_entry']} adx{cfg.get('adx_min')} "
              f"tp_rr{cfg['tp_rr']} sl_pct{cfg.get('sl_pct')} atr_sl{cfg.get('atr_mult_sl', 0)} "
              f"session_peak={cfg.get('session_peak', False)} comm={COMM[sym]*1e4:.1f}bp", flush=True)
        tr = run(sym, cfg, blocked)
        for name, (a, b) in PERIODS.items():
            s = stat([r for t, r in tr if ts(a) <= t < ts(b)])
            print(f"  {name:<14} " + (f"n={s['n']:<5} WR {s['wr']*100:5.1f}%  EV {s['ev']:+.3f}R  "
                                       f"net {s['net']:+7.1f}R  t={s['t']:+.1f}" if s else "too few trades"))
        yrs = []
        for y in range(2018, 2027):
            s = stat([r for t, r in tr if ts(f"{y}-01-01") <= t < ts(f"{y+1}-01-01")])
            yrs.append(f"{y % 100:02d}:{s['ev']:+.2f}" if s else f"{y % 100:02d}:--")
        print("  by year EV: " + "  ".join(yrs))


if __name__ == "__main__":
    sys.exit(main())
