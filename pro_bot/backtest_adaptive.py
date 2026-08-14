"""
Adaptive RSI vs Fixed RSI -- side-by-side comparison on holdout data.

Runs the 3 production configs (XAUUSD, EURUSD, USDJPY) in both modes and
prints a clean comparison table so we can see whether adaptive adds value.

Usage:
  python pro_bot/backtest_adaptive.py
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import (
    load_data, split_bars, simulate_exits, calc_metrics,
    SPREADS, MIN_TRADES,
)
from pro_bot.indicators import ema as _ema, rsi as _rsi, adx as _adx, atr as _atr
from pro_bot.strategies.base import Signal


# ── Production configs (mirrors config.yaml exactly) ─────────────────────────

PROD = {
    "frxXAUUSD": {
        "ema_period":       100,
        "slope_bars":       3,
        "rsi_period":       14,
        "rsi_entry":        35.0,
        "tp_rr":            1.5,
        "adx_min":          25,
        "session_peak":     True,
        "macro_filter":     True,
        "macro_ema_period": 20,
        "be_r":             1.0,
    },
    "frxEURUSD": {
        "ema_period":       200,
        "slope_bars":       3,
        "rsi_period":       14,
        "rsi_entry":        35.0,
        "tp_rr":            2.0,
        "adx_min":          25,
        "macro_filter":     True,
        "macro_ema_period": 50,
        "be_r":             1.0,
    },
    "frxUSDJPY": {
        "ema_period":       100,
        "slope_bars":       3,
        "rsi_period":       14,
        "rsi_entry":        35.0,
        "tp_rr":            2.0,
        "adx_min":          20,
        "atr_mult_sl":      1.5,
        "macro_filter":     True,
        "macro_ema_period": 20,
        "be_r":             1.0,
    },
}


# ── Strategy runner with optional adaptive RSI ────────────────────────────────

def run_mtf(bars_5m, bars_1h, bars_1d=None, config=None, adaptive=False) -> list[tuple]:
    cfg          = config or {}
    ema_p        = cfg.get("ema_period",       200)
    slope_b      = cfg.get("slope_bars",         3)
    rsi_p        = cfg.get("rsi_period",        14)
    rsi_entry    = cfg.get("rsi_entry",        35.0)
    tp_rr        = cfg.get("tp_rr",            2.0)
    adx_min      = cfg.get("adx_min",            0)
    sess_peak    = cfg.get("session_peak",    False)
    atr_mult     = cfg.get("atr_mult_sl",      0.0)
    macro_filter = cfg.get("macro_filter",    False)
    macro_ema_p  = cfg.get("macro_ema_period",  20)
    rsi_lookback = cfg.get("rsi_lookback",      50)

    closes_1h = [b["close"] for b in bars_1h]
    closes_5m = [b["close"] for b in bars_5m]
    ema_1h    = _ema(closes_1h, ema_p)
    rsi_5m    = _rsi(closes_5m, rsi_p)
    adx_1h    = _adx(bars_1h, 14) if adx_min > 0 else None
    atr_5m    = _atr(bars_5m, 14) if atr_mult > 0 else None

    if macro_filter and bars_1d:
        closes_1d = [b["close"] for b in bars_1d]
        ema_1d    = _ema(closes_1d, macro_ema_p)
        epochs_1d = [b["epoch"] for b in bars_1d]
    else:
        ema_1d = epochs_1d = None

    epochs_1h = [b["epoch"] for b in bars_1h]

    results = []
    for i in range(rsi_p + 2, len(bars_5m)):
        if sess_peak:
            h_utc  = (bars_5m[i]["epoch"] % 86400) // 3600
            m_min  = (bars_5m[i]["epoch"] % 3600)  // 60
            london = (7 <= h_utc < 10) or (h_utc == 10 and m_min < 30)
            ny     = (13 <= h_utc < 16) or (h_utc == 16 and m_min < 30)
            if not (london or ny):
                continue

        j = bisect.bisect_right(epochs_1h, bars_5m[i]["epoch"]) - 1
        if j < slope_b or j < 0:
            continue

        e_now  = ema_1h[j]
        e_prev = ema_1h[j - slope_b]
        r_now  = rsi_5m[i]
        r_prev = rsi_5m[i - 1]
        if any(x is None for x in [e_now, e_prev, r_now, r_prev]):
            continue

        if adx_min > 0 and adx_1h is not None:
            adx_val = adx_1h[j]
            if adx_val is None or adx_val < adx_min:
                continue

        allow_long = allow_short = True
        if ema_1d is not None:
            k = bisect.bisect_right(epochs_1d, bars_5m[i]["epoch"]) - 1
            if k < 1 or ema_1d[k] is None or ema_1d[k - 1] is None:
                continue
            macro_up    = ema_1d[k] > ema_1d[k - 1]
            allow_long  = macro_up
            allow_short = not macro_up

        # ── Entry threshold: fixed or adaptive ───────────────────────────────
        if adaptive and i >= rsi_lookback:
            recent = [v for v in rsi_5m[i - rsi_lookback:i] if v is not None]
            if len(recent) >= max(10, rsi_lookback // 2):
                recent.sort()
                idx       = max(0, int(len(recent) * 0.20) - 1)
                threshold = max(rsi_entry, min(50.0, recent[idx]))
            else:
                threshold = rsi_entry
        else:
            threshold = rsi_entry

        ob = 100.0 - threshold

        trend_up   = e_now > e_prev
        trend_down = e_now < e_prev
        price = closes_5m[i]

        if atr_mult > 0 and atr_5m is not None and atr_5m[i] is not None:
            sl = atr_5m[i] * atr_mult
        else:
            sl = abs(price - bars_5m[i]["low"]) + price * 0.0002
        if sl <= 0:
            sl = price * 0.003

        if trend_up and r_prev >= threshold > r_now and allow_long:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=sl * tp_rr)))
        elif trend_down and r_prev <= ob < r_now and allow_short:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=sl * tp_rr)))

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarise(trades, spread_label="") -> dict:
    closed  = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    wins    = [t for t in closed if t.result == "WIN"]
    losses  = [t for t in closed if t.result == "LOSS"]
    be_c    = [t for t in closed if t.result == "BE"]
    n_wr    = len(wins) + len(losses)
    wr      = len(wins) / n_wr if n_wr > 0 else 0.0
    ev      = sum(t.r_multiple for t in closed) / len(closed) if closed else 0.0
    net_r   = sum(t.r_multiple for t in closed)
    # per-day
    if len(trades) >= 2:
        from pro_bot.backtest import GRAN_5M
        span_bars = trades[-1].signal_bar - trades[0].signal_bar
        days      = span_bars * GRAN_5M / 86400
        per_day   = len(closed) / days if days > 0 else 0
    else:
        per_day = 0
    return dict(closed=len(closed), wins=len(wins), losses=len(losses),
                be=len(be_c), wr=wr, ev=ev, net_r=net_r, per_day=per_day)


def _row(label, s) -> str:
    verdict = ("STRONG"  if s["ev"] > 0.05
               else "OK" if s["ev"] > 0
               else "NEG")
    return (f"  {label:<38}  "
            f"trades={s['closed']:>3}({s['per_day']:.1f}/d)  "
            f"W:{s['wins']} L:{s['losses']} BE:{s['be']}  "
            f"WR:{s['wr']*100:>5.1f}%  EV:{s['ev']:>+.3f}R  "
            f"Net:{s['net_r']:>+6.1f}R  [{verdict}]")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 80)
    print("ADAPTIVE RSI vs FIXED RSI  —  holdout validation (20% unseen data)")
    print("Spread + BE applied.  MIN_TRADES for full metrics not enforced here.")
    print("=" * 80)

    for sym, cfg in PROD.items():
        spread = SPREADS.get(sym, 0.0)
        be_r   = cfg.get("be_r", 0.0)

        print(f"\n{'─'*80}")
        print(f"  {sym}  |  spread={spread}  BE@{be_r}R  "
              f"RSI_entry={cfg['rsi_entry']}  RR={cfg['tp_rr']}")
        print(f"{'─'*80}")

        try:
            bars_5m, bars_1h, bars_1d = await load_data(sym)
        except Exception as e:
            print(f"  SKIP — {e}")
            continue

        _, holdout = split_bars(bars_5m, bars_1h, bars_1d)
        h5  = holdout["5m"]
        h1h = holdout["1h"]
        h1d = holdout["1d"] if cfg.get("macro_filter") else None

        print(f"  Holdout: {len(h5)} x 5min bars")

        for mode_label, adaptive in [
            (f"FIXED   RSI<{cfg['rsi_entry']:.0f}", False),
            ("ADAPTIVE 20th-pct (lookback 50)",  True),
        ]:
            sigs   = run_mtf(h5, h1h, h1d, cfg, adaptive=adaptive)
            trades = simulate_exits(h5, sigs, spread=spread, be_r=be_r)
            s      = _summarise(trades)
            print(_row(mode_label, s))

    print(f"\n{'='*80}")
    print("Interpretation:")
    print("  EV > 0.05R = STRONG edge  |  EV > 0 = OK  |  EV <= 0 = NEG")
    print("  If ADAPTIVE EV >= FIXED EV → adaptive worth keeping")
    print("  If ADAPTIVE EV <  FIXED EV → adaptive hurts, revert rsi_adaptive: false")
    print()


if __name__ == "__main__":
    asyncio.run(main())
