"""
Portfolio-level entry filters for the enabled XAUUSD research strategies (pre-registered 2026-11-10).

Question: is there a generic filter that, applied identically to every strategy, removes
worse-than-average trades out of sample? Per-strategy filters are not tested (too many
choices -> curve fitting).

Trades: every signal of each enabled H1/M15 XAUUSD strategy (ofbs excluded, mtf_pullback
excluded — M5 history too short), replayed with the live strategy classes and the
SL/TP simulator from backtest_xau_oos.py (cost 0.30 at $4000, 5-day max hold).

Filters (feature measured at the signal bar close, using only completed daily bars):
  TREND   trade direction agrees with close vs daily EMA50
  MOM20   trade direction agrees with the 20-day return
  NOVOL   skip if daily ATR14 is above its 90th percentile of the past 250 days
  NONEWS  skip entries 08:00-10:30 New York time (US data releases)
  NOASIA  skip entries 22:00-07:00 UTC
plus every pair of them (15 rule sets).

Split: TRAIN 2016-07 -> 2020-12 | TEST 2021-01 -> now.
Pass rule (fixed before running): in BOTH periods the kept trades beat the removed trades
(pooled, Welch t >= 2 on TRAIN and >= 1.5 on TEST), the filter raises EV in >= 2/3 of
strategies, and pooled kept EV > 0 on TEST.

    python pro_bot/backtest_portfolio_filters.py
"""

import itertools
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from pro_bot.backtest_xau_oos import MAX_HOLD, TF_NAME, load, run, simulate, ts  # noqa: E402
from pro_bot.incubator import server_offset  # noqa: E402
from pro_bot.main import STRATEGY_CLASSES  # noqa: E402

START = "2016-07-01"
TEST_START = ts("2021-01-01")
NY = ZoneInfo("America/New_York")
EXCLUDE = {"ofbs", "mtf_pullback"}


def daily_features(d1):
    closes = np.array([b["close"] for b in d1])
    highs = np.array([b["high"] for b in d1])
    lows = np.array([b["low"] for b in d1])
    ema = np.empty_like(closes)
    a = 2 / 51
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = a * closes[i] + (1 - a) * ema[i - 1]
    tr = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
    tr = np.r_[highs[0] - lows[0], tr]
    atr = np.convolve(tr, np.ones(14) / 14, "full")[:len(tr)]
    atr_pct = np.full(len(atr), np.nan)
    for i in range(264, len(atr)):
        win = atr[i - 250:i + 1]
        atr_pct[i] = (win < atr[i]).mean()
    mom20 = np.r_[np.full(20, np.nan), closes[20:] / closes[:-20] - 1]
    ends = np.array([b["epoch"] + 86400 for b in d1])
    return ends, closes, ema, atr_pct, mom20


def features_for(trade_epoch, gran, direction, dfeat):
    ends, closes, ema, atr_pct, mom20 = dfeat
    close_t = trade_epoch + gran
    k = np.searchsorted(ends, close_t, "right") - 1          # last completed daily bar
    utc = close_t - server_offset(close_t)
    dt_utc = datetime.fromtimestamp(utc, timezone.utc)
    ny = dt_utc.astimezone(NY)
    ny_min = ny.hour * 60 + ny.minute
    return {
        "TREND": k >= 50 and np.sign(closes[k] - ema[k]) == direction,
        "MOM20": k >= 20 and np.sign(mom20[k]) == direction,
        "NOVOL": not (k >= 0 and atr_pct[k] > 0.90),
        "NONEWS": not (ny.weekday() < 5 and 8 * 60 <= ny_min < 10 * 60 + 30),
        "NOASIA": not (dt_utc.hour >= 22 or dt_utc.hour < 7),
    }


def welch_t(a, b):
    if len(a) < 5 or len(b) < 5:
        return 0.0
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    return (np.mean(a) - np.mean(b)) / math.sqrt(va + vb) if va + vb > 0 else 0.0


def main():
    cfg_all = yaml.safe_load(open(ROOT / "pro_bot" / "config.yaml", encoding="utf-8"))
    strats = [s for s in cfg_all["strategies"]
              if s.get("enabled", True) and "XAUUSD" in s.get("symbols", [])
              and s.get("granularity") in (3600, 900) and s["name"] not in EXCLUDE]
    d1 = load("D1")
    dfeat = daily_features(d1)
    series = {g: load(TF_NAME[g]) for g in {s["granularity"] for s in strats}}

    trades = []          # (strategy, epoch, R, features)
    for s in strats:
        g = s["granularity"]
        bars = series[g]
        sigs = run(STRATEGY_CLASSES[s["name"]], s, bars, d1, g, ts(START))
        dirs = {bars[i]["epoch"]: (1 if sig.action == "BUY" else -1) for i, sig in sigs}
        res = simulate(bars, sigs, MAX_HOLD[g])
        for e, r in res:
            trades.append((s["name"], e, r, features_for(e, g, dirs[e], dfeat)))
        n_tr = sum(1 for e, _ in res if e < TEST_START)
        print(f"{s['name']:<5} {TF_NAME[g]}: {len(res)} trades ({n_tr} train / {len(res) - n_tr} test)", flush=True)

    names = sorted({t[0] for t in trades})
    base_filters = ["TREND", "MOM20", "NOVOL", "NONEWS", "NOASIA"]
    rule_sets = [(f,) for f in base_filters] + list(itertools.combinations(base_filters, 2))

    def evaluate(rule, part):
        sel = [t for t in trades if (t[1] < TEST_START) == (part == "train")]
        keep = [t for t in sel if all(t[3][f] for f in rule)]
        drop = [t for t in sel if not all(t[3][f] for f in rule)]
        rk, rd, ra = (np.array([t[2] for t in x]) for x in (keep, drop, sel))
        improved, counted = 0, 0
        for nm in names:
            a = [t[2] for t in sel if t[0] == nm]
            k = [t[2] for t in keep if t[0] == nm]
            if len(a) >= 20 and len(k) >= 10:
                counted += 1
                improved += np.mean(k) > np.mean(a)
        return dict(n_all=len(ra), ev_all=ra.mean() if len(ra) else np.nan,
                    n_keep=len(rk), ev_keep=rk.mean() if len(rk) else np.nan,
                    ev_drop=rd.mean() if len(rd) else np.nan, t=welch_t(rk, rd),
                    improved=improved, counted=counted)

    print("\n=== BASELINE per strategy (EV in R) ===")
    for nm in names:
        row = []
        for part, cond in (("train", lambda e: e < TEST_START), ("test", lambda e: e >= TEST_START)):
            rs = np.array([t[2] for t in trades if t[0] == nm and cond(t[1])])
            row.append(f"{part} n={len(rs):4d} EV={rs.mean() if len(rs) else float('nan'):+.3f}")
        print(f"  {nm:<5} " + " | ".join(row))
    for part in ("train", "test"):
        r = evaluate((), part)
        print(f"  POOLED {part}: n={r['n_all']} EV={r['ev_all']:+.3f}R")

    print("\n=== FILTERS (kept vs removed; improved = strategies whose EV rises) ===")
    print(f"{'rule':<16} | {'TRAIN keep% EVkeep EVdrop t  improved':<44} | {'TEST keep% EVkeep EVdrop t  improved':<44} | verdict")
    for rule in rule_sets:
        a, b = evaluate(rule, "train"), evaluate(rule, "test")
        ok = (a["t"] >= 2 and b["t"] >= 1.5 and b["ev_keep"] > 0
              and a["improved"] * 3 >= 2 * a["counted"] and b["improved"] * 3 >= 2 * b["counted"])

        def c(x):
            return (f"{x['n_keep'] / max(x['n_all'], 1) * 100:4.0f}% {x['ev_keep']:+.3f} {x['ev_drop']:+.3f} "
                    f"{x['t']:+5.2f} {x['improved']}/{x['counted']}")
        print(f"{'+'.join(rule):<16} | {c(a):<44} | {c(b):<44} | {'PASS' if ok else 'fail'}")

    print("\n=== POST-HOC yearly view (inspection only, no re-tuning): EV kept - EV removed, R ===")
    years = sorted({datetime.fromtimestamp(t[1], timezone.utc).year for t in trades})
    for rule in [("NOVOL",), ("TREND",), ("TREND", "NOVOL")]:
        cells = []
        for y in years:
            sel = [t for t in trades if datetime.fromtimestamp(t[1], timezone.utc).year == y]
            k = [t[2] for t in sel if all(t[3][f] for f in rule)]
            d = [t[2] for t in sel if not all(t[3][f] for f in rule)]
            cells.append(f"{y % 100:02d}:{np.mean(k) - np.mean(d):+.2f}({len(d)})" if len(d) >= 10 else f"{y % 100:02d}:--")
        print(f"{'+'.join(rule):<12} " + " ".join(cells))
    base = []
    for y in years:
        rs = [t[2] for t in trades if datetime.fromtimestamp(t[1], timezone.utc).year == y]
        base.append(f"{y % 100:02d}:{np.mean(rs):+.3f}(n={len(rs)})")
    print("baseline EV  " + " ".join(base))


if __name__ == "__main__":
    main()
