"""
Out-of-sample replay of every ENABLED XAUUSD research strategy (H1 / M15) in
pro_bot/config.yaml, using the live strategy classes and MT5 broker history.

The strategies were tuned on <= 2 years of Deriv data ending Aug 2026, so:
  OOS-A  2010-01 -> 2024-07   (never seen)
  IS     2024-08 -> 2026-08   (tuning window, for comparison only)
  OOS-B  2026-08 -> today     (never seen)

Bars are fed exactly like pro_bot/main.py's research handler: completed daily
bars via feed_daily(), then each completed H1/M15 bar via feed(). Every signal
is taken (the live one-position-per-symbol limit hides most signals).
Exit: signal SL/TP from entry at bar close +/- cost, SL wins same-bar ties,
mark-to-market after MAX_HOLD bars. Cost scales with price (0.30 at $4000 =
IC Markets raw spread + commission).

Usage:
    python pro_bot/backtest_xau_oos.py [--refresh]
"""

import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from pro_bot.main import STRATEGY_CLASSES  # noqa: E402

DATA = ROOT / "data" / "pro"
COST_FRAC = 0.30 / 4000
MAX_HOLD = {3600: 120, 900: 192}            # 5 days
TF_NAME = {3600: "H1", 900: "M15", 86400: "D1"}

OOS_A = ("2010-01-01", "2024-08-01")
IS = ("2024-08-01", "2026-08-04")
OOS_B = ("2026-08-04", "2100-01-01")


def ts(s):
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def load(tf_name, refresh=False):
    p = DATA / f"mt5_XAUUSD_{tf_name}.csv"
    if refresh or not p.exists():
        import MetaTrader5 as mt5
        assert mt5.initialize(), mt5.last_error()
        tf = {"M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}[tf_name]
        rates = mt5.copy_rates_from_pos("XAUUSD", tf, 0, mt5.terminal_info().maxbars - 1)
        mt5.shutdown()
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "open", "high", "low", "close", "tick_volume"])
            for r in rates[:-1]:
                w.writerow([int(r["time"]), r["open"], r["high"], r["low"], r["close"], int(r["tick_volume"])])
    with open(p, newline="") as f:
        return [{"epoch": int(r["epoch"]), "open": float(r["open"]), "high": float(r["high"]),
                 "low": float(r["low"]), "close": float(r["close"]), "volume": int(r["tick_volume"]),
                 "tick_volume": int(r["tick_volume"])} for r in csv.DictReader(f)]


def run(cls, cfg, bars, d1, gran, start):
    st = cls(dict(cfg))
    di, sigs = 0, []
    seed = 299                                   # live seeds history_count-1 bars
    while di < len(d1) and d1[di]["epoch"] + 86400 <= bars[seed]["epoch"]:
        st.feed_daily(d1[di]); di += 1
    st._seed_h1(bars[:seed])
    for i, bar in enumerate(bars):
        if i < seed:
            continue
        close_t = bar["epoch"] + gran
        while di < len(d1) and d1[di]["epoch"] + 86400 <= close_t:
            st.feed_daily(d1[di]); di += 1
        sig = st.feed(bar)
        if sig.action != "HOLD" and bar["epoch"] >= start:
            sigs.append((i, sig))
    return sigs


def simulate(bars, sigs, max_hold):
    out = []
    for idx, sig in sigs:
        sl_d, tp_d = sig.sl_pips, sig.tp_pips
        if not sl_d or sl_d <= 0:
            continue
        c = bars[idx]["close"]
        cost = c * COST_FRAC
        d = 1 if sig.action == "BUY" else -1
        entry = c + d * cost
        sl, tp = entry - d * sl_d, entry + d * tp_d
        r = None
        last = min(idx + max_hold, len(bars) - 1)
        for j in range(idx + 1, last + 1):
            h, l = bars[j]["high"], bars[j]["low"]
            if (d == 1 and l <= sl) or (d == -1 and h >= sl):
                r = -1.0
            elif (d == 1 and h >= tp) or (d == -1 and l <= tp):
                r = tp_d / sl_d
            if r is not None:
                break
        if r is None:
            r = d * (bars[last]["close"] - entry) / sl_d
        out.append((bars[idx]["epoch"], r))
    return out


def st(rs):
    n = len(rs)
    if n == 0:
        return None
    ev = sum(rs) / n
    sd = math.sqrt(sum((x - ev) ** 2 for x in rs) / (n - 1)) if n > 1 else 0
    return dict(n=n, ev=ev, net=sum(rs), t=ev / (sd / math.sqrt(n)) if sd else 0)


def cell(s):
    return f"{'--':>22}" if not s else f"{s['n']:>5} {s['ev']:>+6.3f} t={s['t']:>+5.1f}"


def main():
    refresh = "--refresh" in sys.argv
    cfg_all = yaml.safe_load(open(ROOT / "pro_bot" / "config.yaml", encoding="utf-8"))
    strats = [s for s in cfg_all["strategies"]
              if s.get("enabled", True) and "XAUUSD" in s.get("symbols", [])
              and s.get("granularity") in (3600, 900) and "ltf_granularity" not in s]

    d1 = load("D1", refresh)
    series = {g: load(TF_NAME[g], refresh) for g in {s["granularity"] for s in strats}}
    f = lambda e: datetime.fromtimestamp(e, timezone.utc).strftime("%Y-%m-%d")
    for g, b in series.items():
        print(f"{TF_NAME[g]}: {len(b)} bars {f(b[0]['epoch'])} -> {f(b[-1]['epoch'])}")

    periods = {"OOS-A 2010-24": OOS_A, "IS 2024-26": IS, "OOS-B Aug26+": OOS_B}
    blocks = [(y, y + 3) for y in range(2010, 2025, 3)]

    print(f"\n{'strategy':<6} {'tf':<4} {'cfg RR':>6} | " +
          " | ".join(f"{k:^22}" for k in periods) + " | 3-yr blocks EV (OOS-A)")
    print("-" * 150)
    verdicts = []
    for s in strats:
        g = s["granularity"]
        bars = series[g]
        sigs = run(STRATEGY_CLASSES[s["name"]], s, bars, d1, g, ts(OOS_A[0]))
        trades = simulate(bars, sigs, MAX_HOLD[g])
        per = {k: st([r for e, r in trades if ts(a) <= e < ts(b)]) for k, (a, b) in periods.items()}
        blk = []
        pos_blocks = 0
        for y0, y1 in blocks:
            bs = st([r for e, r in trades if ts(f"{y0}-01-01") <= e < min(ts(f"{y1}-01-01"), ts(OOS_A[1]))])
            if bs:
                pos_blocks += bs["ev"] > 0
                blk.append(f"{y0 % 100:02d}:{bs['ev']:+.2f}")
            else:
                blk.append(f"{y0 % 100:02d}:--")
        print(f"{s['name']:<6} {TF_NAME[g]:<4} {s['tp_rr']:>6} | " +
              " | ".join(cell(per[k]) for k in periods) + " | " + " ".join(blk))

        a, b_ = per["OOS-A 2010-24"], per["OOS-B Aug26+"]
        nb = sum(1 for x in blk if not x.endswith("--"))
        if a and a["ev"] > 0 and a["t"] >= 2 and pos_blocks >= max(1, nb - 1):
            v = "KEEP"
        elif a and a["ev"] > 0 and pos_blocks * 2 >= nb:
            v = "WEAK"
        else:
            v = "DROP"
        verdicts.append((s["name"], v, a, b_, pos_blocks, nb))

    print("\nVerdict (KEEP: OOS-A EV>0 with t>=2 and positive in all but <=1 three-year block;"
          " WEAK: EV>0 but not significant/consistent; DROP: otherwise)")
    for name, v, a, b_, pb, nb in verdicts:
        print(f"  {name:<6} {v:<5} OOS-A {a['ev'] if a else float('nan'):+.3f}R n={a['n'] if a else 0:<5} "
              f"blocks+ {pb}/{nb}   OOS-B {b_['ev'] if b_ else float('nan'):+.3f}R n={b_['n'] if b_ else 0}")


if __name__ == "__main__":
    main()
