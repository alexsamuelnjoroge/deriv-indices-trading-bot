"""
Long-history test of pro_bot's live mtf_pullback strategy (XAUUSD 5-min) on
Dukascopy data 2018 -> now, instead of MT5's 17 months.

The live strategy class is driven exactly like pro_bot/backtest_mtf_filters.py
(which mirrors main.py's MTF handler). Dukascopy is UTC; the live bot sees IC
Markets server time (New York + 7h = UTC+2 winter / UTC+3 summer), so epochs
are shifted to server time and H1/H4/D1 bars are built on server-time
boundaries like MT5.

Costs: 0.30 USD/oz per round trip (raw spread + commission). Conservative
same-bar fills, open trades marked to market after 24h. Overnight swap is not
included (most trades close intraday) — noted as a caveat.

Split: DEV 2018-2021 | VAL 2022 -> 2024-07 | RECENT 2024-08 -> now.
"""

import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pro_bot.backtest_mtf_filters import load_cfg, run_strategy, simulate, stats  # noqa: E402

COST = 0.30
PER = {"DEV": ("2018-01-01", "2022-01-01"), "VAL": ("2022-01-01", "2024-08-01"),
       "RECENT": ("2024-08-01", "2100-01-01")}


def load_server_time() -> dict:
    parts = [pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(glob.glob(str(ROOT / "data/duka/frxXAUUSD_M5_*.csv")))]
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    utc = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    ny_off = utc.tz_convert("America/New_York").map(lambda t: t.utcoffset().total_seconds())
    server = utc.tz_localize(None) + pd.to_timedelta(np.asarray(ny_off) + 7 * 3600, unit="s")
    df.index = server
    df["tick_volume"] = 1

    def bars(frame: pd.DataFrame) -> list[dict]:
        ep = ((frame.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).astype(np.int64)
        return [{"epoch": int(e), "open": o, "high": h, "low": l, "close": c, "tick_volume": 1}
                for e, o, h, l, c in zip(ep, frame["open"], frame["high"], frame["low"], frame["close"])]

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = {"M5": bars(df)}
    for name, rule in (("H1", "1h"), ("H4", "4h"), ("D1", "1D")):
        out[name] = bars(df[["open", "high", "low", "close"]].resample(rule, label="left", closed="left").agg(agg).dropna())
    return out


def main():
    data = load_server_time()
    m5 = data["M5"]
    f = lambda e: pd.Timestamp(e, unit="s").strftime("%Y-%m-%d")
    print(f"Dukascopy XAUUSD in server time: M5 {len(m5):,} bars {f(m5[0]['epoch'])} -> {f(m5[-1]['epoch'])}"
          f" | H1 {len(data['H1']):,} | H4 {len(data['H4']):,} | D1 {len(data['D1']):,}", flush=True)
    cfg = load_cfg()
    print(f"config: ema{cfg['ema_period']} rsi_entry{cfg['rsi_entry']} adaptive={cfg.get('rsi_adaptive')} "
          f"adx{cfg['adx_min']} tp_rr{cfg['tp_rr']} atr_sl{cfg['atr_mult_sl']} 4h={cfg.get('use_4h_filter')} "
          f"session_peak={cfg.get('session_peak')} cost {COST}\n", flush=True)

    sigs = run_strategy(data, cfg)
    trades = simulate(m5, sigs, spread=COST)
    ts = lambda s: int((pd.Timestamp(s) - pd.Timestamp(0)) // pd.Timedelta(seconds=1))
    print(f"{'period':<8} {'n':>5} {'WR':>6} {'EV R':>8} {'t':>6} {'net R':>8} {'maxDD R':>8}  {'EV one-at-a-time':>16}")
    for name, (a, b) in PER.items():
        tr = [t for t in trades if ts(a) <= t["epoch"] < ts(b)]
        s = stats(tr)
        if not s["n"]:
            print(f"{name:<8} no trades"); continue
        tval = s["ev"] / s["se"] if s["se"] else 0
        print(f"{name:<8} {s['n']:>5} {s['wr']*100:5.1f}% {s['ev']:+8.3f} {tval:+6.1f} {s['net']:+8.1f} {s['dd']:8.1f}"
              f"  {s['ev_seq']:+8.3f} (n={s['n_seq']})")
    print("\nby year:")
    for y in range(2018, 2027):
        tr = [t for t in trades if ts(f"{y}-01-01") <= t["epoch"] < ts(f"{y+1}-01-01")]
        s = stats(tr)
        if s["n"]:
            print(f"  {y}: n={s['n']:>4}  WR {s['wr']*100:5.1f}%  EV {s['ev']:+.3f}R  net {s['net']:+6.1f}R")
    for side in ("BUY", "SELL"):
        s = stats([t for t in trades if t["side"] == side])
        print(f"{side}: n={s['n']}  EV {s['ev']:+.3f}R  WR {s['wr']*100:.1f}%")
    print(f"\n(breakeven WR at {cfg['tp_rr']}R = {100/(1+cfg['tp_rr']):.1f}%; swap not included)")


if __name__ == "__main__":
    main()
