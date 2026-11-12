"""
Macro-gate variants for the enabled XAUUSD research strategies (pre-registered 2026-11-10).

The shared gate (ResearchDailyStrategy._macro_gate) allows only BUYs when the daily EMA20
rises and only SELLs when it falls — no neutral zone. Variants:
  V0 current        sign of the EMA20 slope
  V1 band 0.01%     |slope| < 0.01% of EMA -> both directions allowed
  V2 band 0.05%     |slope| < 0.05% of EMA -> both directions allowed
  V3 flat=stand by  |slope| < 0.05% of EMA -> no trades
  V4 no gate        both directions always

Trades: live strategy classes replayed with the backtest_xau_oos.py simulator (ofbs and
mtf_pullback excluded). The strategies were tuned on Aug 2024 - Aug 2026 data, so the
judging window is CLEAN = 2016-07 -> 2024-07 (halves: 2016-07..2020 and 2021..2024-07);
2024-08 -> now is reported but is in-sample.
A variant beats V0 only if pooled EV and total R/yr are higher in BOTH clean halves and
EV improves in >= 2/3 of strategies over the full clean window.

    python pro_bot/backtest_macro_gate.py
"""

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from pro_bot.backtest_xau_oos import MAX_HOLD, TF_NAME, load, run, simulate, ts  # noqa: E402
from pro_bot.main import STRATEGY_CLASSES  # noqa: E402
from pro_bot.strategies.research_base import ResearchDailyStrategy  # noqa: E402

START = "2016-07-01"
PERIODS = {"clean A 2016-20": (ts("2016-07-01"), ts("2021-01-01")),
           "clean B 2021-24": (ts("2021-01-01"), ts("2024-08-01")),
           "in-sample 24-26": (ts("2024-08-01"), ts("2100-01-01"))}
CLEAN = (ts("2016-07-01"), ts("2024-08-01"))
EXCLUDE = {"ofbs", "mtf_pullback"}


def ema_pair(self):
    d1 = list(self._d1)
    p = self.macro_ema_period
    if len(d1) < p + 1:
        return None
    closes = [b["close"] for b in d1]
    k = 2 / (p + 1)
    ema = sum(closes[:p]) / p
    prev = ema
    for c in closes[p:]:
        prev = ema
        ema = c * k + ema * (1 - k)
    return ema, prev


def make_gate(kind, band=0.0):
    def gate(self):
        if kind == "none" or not self.macro_filter:
            return True, True
        ep = ema_pair(self)
        if ep is None:
            return True, True
        ema, prev = ep
        flat = abs(ema - prev) < band * ema
        if kind == "band" and flat:
            return True, True
        if kind == "standby" and flat:
            return False, False
        return ema > prev, ema <= prev
    return gate


VARIANTS = {
    "V0 current": make_gate("sign"),
    "V1 band 0.01%": make_gate("band", 0.0001),
    "V2 band 0.05%": make_gate("band", 0.0005),
    "V3 flat=stand by": make_gate("standby", 0.0005),
    "V4 no gate": make_gate("none"),
}


def main():
    cfg_all = yaml.safe_load(open(ROOT / "pro_bot" / "config.yaml", encoding="utf-8"))
    strats = [s for s in cfg_all["strategies"]
              if s.get("enabled", True) and "XAUUSD" in s.get("symbols", [])
              and s.get("granularity") in (3600, 900) and s["name"] not in EXCLUDE]
    d1 = load("D1")
    series = {g: load(TF_NAME[g]) for g in {s["granularity"] for s in strats}}

    results = {}
    original = ResearchDailyStrategy._macro_gate
    for vname, gate in VARIANTS.items():
        ResearchDailyStrategy._macro_gate = gate
        trades = []
        for s in strats:
            g = s["granularity"]
            bars = series[g]
            sigs = run(STRATEGY_CLASSES[s["name"]], s, bars, d1, g, ts(START))
            side = {bars[i]["epoch"]: (1 if sig.action == "BUY" else -1) for i, sig in sigs}
            trades += [(s["name"], e, r, side[e]) for e, r in simulate(bars, sigs, MAX_HOLD[g])]
        results[vname] = trades
        print(f"{vname}: {len(trades)} trades", flush=True)
    ResearchDailyStrategy._macro_gate = original

    def summ(trades, a, b):
        rs = np.array([t[2] for t in trades if a <= t[1] < b])
        if len(rs) < 10:
            return None
        yrs = (b - a if b < ts("2100-01-01") else datetime.now(timezone.utc).timestamp() - a) / 86400 / 365.25
        sd = rs.std(ddof=1)
        longs = sum(1 for t in trades if a <= t[1] < b and t[3] > 0)
        return dict(n=len(rs), ev=rs.mean(), t=rs.mean() / sd * math.sqrt(len(rs)) if sd else 0,
                    ryr=rs.sum() / yrs, long_pct=longs / len(rs) * 100)

    print("\n=== Pooled, all strategies (EV = mean R per trade; R/yr = total R per year) ===")
    for vname, trades in results.items():
        cells = []
        for pname, (a, b) in PERIODS.items():
            s = summ(trades, a, b)
            cells.append(f"{pname}: n={s['n']:4d} EV={s['ev']:+.3f} t={s['t']:+5.2f} R/yr={s['ryr']:+6.1f} long={s['long_pct']:3.0f}%")
        print(f"{vname:<17} " + " | ".join(cells))

    base = results["V0 current"]
    names = sorted({t[0] for t in base})

    def strat_ev(trades, nm, a, b):
        rs = [t[2] for t in trades if t[0] == nm and a <= t[1] < b]
        return np.mean(rs) if len(rs) >= 15 else None

    print("\n=== Per strategy, CLEAN 2016-07 -> 2024-07 EV (R) ===")
    print(f"{'strategy':<6} " + " ".join(f"{v[:16]:>17}" for v in VARIANTS))
    for nm in names:
        print(f"{nm:<6} " + " ".join(
            f"{strat_ev(results[v], nm, *CLEAN):>+17.3f}" if strat_ev(results[v], nm, *CLEAN) is not None
            else f"{'--':>17}" for v in VARIANTS))

    print("\n=== Verdict vs V0 ===")
    for vname, trades in results.items():
        if vname == "V0 current":
            continue
        better_halves = True
        for pname in ("clean A 2016-20", "clean B 2021-24"):
            a, b = PERIODS[pname]
            s, s0 = summ(trades, a, b), summ(base, a, b)
            better_halves &= s["ev"] > s0["ev"] and s["ryr"] > s0["ryr"]
        imp = cnt = 0
        for nm in names:
            e, e0 = strat_ev(trades, nm, *CLEAN), strat_ev(base, nm, *CLEAN)
            if e is not None and e0 is not None:
                cnt += 1
                imp += e > e0
        ok = better_halves and imp * 3 >= 2 * cnt
        print(f"{vname:<17} better in both clean halves: {better_halves} | strategies improved {imp}/{cnt} | "
              f"{'BETTER' if ok else 'not better'}")


if __name__ == "__main__":
    main()
