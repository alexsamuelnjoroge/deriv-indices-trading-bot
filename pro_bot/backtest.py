"""
Pro Bot Backtest Engine

Simulates each strategy with dynamic SL/TP exits on historical OHLC data.
Unlike binary options research (fixed expiry), here:
  WIN  = price reaches TP before SL  → profit of tp_pips
  LOSS = price reaches SL before TP  → loss of sl_pips

Breakeven WR:
  1:1.5 R:R → 40.0%   (vs 53% for binary)
  1:2.0 R:R → 33.3%   (vs 53% for binary)

Data: reuses cached OHLC from data/scalp/ and data/pro/

Usage:
  python3 pro_bot/backtest.py
  python3 pro_bot/backtest.py --strategy mtf
  python3 pro_bot/backtest.py --symbol frxXAUUSD
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.strategies import (
    MTFPullbackStrategy,
    StochEMAStrategy,
    SessionBreakoutStrategy,
    RSIDivergenceStrategy,
    PivotBounceStrategy,
)
from pro_bot.strategies.base import Signal

LEGACY_WS  = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_5M   = Path("data/scalp")
CACHE_1H   = Path("data/pro")
CACHE_1D   = Path("data/pro")
GRAN_5M    = 300
GRAN_1H    = 3600
GRAN_1D    = 86400

FILTER_STRAT = next((a.split("=")[1] for a in sys.argv if a.startswith("--strategy=")), None)
FILTER_SYM   = next((a.split("=")[1] for a in sys.argv if a.startswith("--symbol=")), None)

SYMBOLS = {
    "frxXAUUSD": {"label": "Gold   ", "pip": 0.01},
    "frxUSDJPY": {"label": "USD/JPY", "pip": 0.01},
    "frxEURUSD": {"label": "EUR/USD", "pip": 0.0001},
    "frxGBPUSD": {"label": "GBP/USD", "pip": 0.0001},
    "frxAUDUSD": {"label": "AUD/USD", "pip": 0.0001},
}


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

async def _fetch(symbol: str, gran: int, count: int, cache_dir: Path) -> list[dict]:
    cache = cache_dir / f"{symbol}_{gran}_ohlc.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    async with websockets.connect(LEGACY_WS, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   gran,
            "count":         count,
            "end":           "latest",
            "req_id":        1,
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)

    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])

    bars = [{"open":  float(c["open"]),
             "high":  float(c["high"]),
             "low":   float(c["low"]),
             "close": float(c["close"]),
             "epoch": int(c["epoch"])} for c in msg["candles"]]
    with open(cache, "w") as f:
        json.dump(bars, f)
    return bars


async def load_data(symbol: str) -> tuple[list[dict], list[dict], list[dict]]:
    bars_5m = await _fetch(symbol, GRAN_5M, 5000, CACHE_5M)
    bars_1h = await _fetch(symbol, GRAN_1H, 2000, CACHE_1H)
    bars_1d = await _fetch(symbol, GRAN_1D,  500, CACHE_1D)
    return bars_5m, bars_1h, bars_1d


# ═══════════════════════════════════════════════════════════════
# SL/TP exit simulator
# ═══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_price: float
    action:      str      # BUY or SELL
    sl:          float    # absolute SL price
    tp:          float    # absolute TP price
    sl_pips:     float
    tp_pips:     float
    signal_bar:  int
    exit_bar:    int  = -1
    result:      str  = ""   # WIN / LOSS / TIMEOUT
    r_multiple:  float = 0.0


def simulate_exits(bars: list[dict], signals: list[tuple],
                   max_hold_bars: int = 48) -> list[Trade]:
    """
    For each signal, scan forward through bars to find SL or TP hit.
    Intrabar logic: if both hit on same bar, assume SL hit first (worst case).
    """
    trades: list[Trade] = []

    for bar_idx, sig in signals:
        if sig.action not in ("BUY", "SELL"):
            continue
        if sig.sl_pips is None or sig.tp_pips is None:
            continue
        if sig.sl_pips <= 0 or sig.tp_pips <= 0:
            continue

        entry = bars[bar_idx]["close"]

        if sig.action == "BUY":
            sl_price = entry - sig.sl_pips
            tp_price = entry + sig.tp_pips
        else:
            sl_price = entry + sig.sl_pips
            tp_price = entry - sig.tp_pips

        trade = Trade(
            entry_price=entry,
            action=sig.action,
            sl=sl_price,
            tp=tp_price,
            sl_pips=sig.sl_pips,
            tp_pips=sig.tp_pips,
            signal_bar=bar_idx,
        )

        for j in range(bar_idx + 1, min(bar_idx + max_hold_bars + 1, len(bars))):
            h = bars[j]["high"]
            l = bars[j]["low"]

            if sig.action == "BUY":
                sl_hit = l <= sl_price
                tp_hit = h >= tp_price
            else:
                sl_hit = h >= sl_price
                tp_hit = l <= tp_price

            # If both hit same bar → SL wins (worst case)
            if sl_hit:
                trade.exit_bar   = j
                trade.result     = "LOSS"
                trade.r_multiple = -1.0
                break
            if tp_hit:
                trade.exit_bar   = j
                trade.result     = "WIN"
                trade.r_multiple = sig.tp_pips / sig.sl_pips
                break
        else:
            trade.exit_bar   = bar_idx + max_hold_bars
            trade.result     = "TIMEOUT"
            trade.r_multiple = 0.0

        trades.append(trade)

    return trades


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def calc_metrics(trades: list[Trade], symbol: str, strategy: str) -> dict:
    closed  = [t for t in trades if t.result in ("WIN", "LOSS")]
    if not closed:
        return {}

    wins    = [t for t in closed if t.result == "WIN"]
    losses  = [t for t in closed if t.result == "LOSS"]
    timeout = [t for t in trades if t.result == "TIMEOUT"]

    wr          = len(wins) / len(closed)
    avg_rr      = (sum(t.r_multiple for t in wins) / len(wins)) if wins else 0
    profit_f    = (len(wins) * avg_rr) / len(losses) if losses else float("inf")
    ev_per_r    = wr * avg_rr - (1 - wr)

    # Equity curve in R
    equity  = 0.0
    peak    = 0.0
    max_dd  = 0.0
    for t in closed:
        equity += t.r_multiple
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Max consecutive losses
    streak = max_streak = 0
    for t in closed:
        if t.result == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    days = (trades[-1].signal_bar - trades[0].signal_bar) * GRAN_5M / 86400
    per_day = len(closed) / days if days > 0 else 0

    return {
        "symbol":      symbol,
        "strategy":    strategy,
        "total":       len(trades),
        "closed":      len(closed),
        "wins":        len(wins),
        "losses":      len(losses),
        "timeouts":    len(timeout),
        "wr":          wr,
        "avg_rr":      avg_rr,
        "profit_f":    profit_f,
        "ev_per_r":    ev_per_r,
        "net_r":       equity,
        "max_dd_r":    max_dd,
        "max_streak":  max_streak,
        "per_day":     per_day,
    }


def print_metrics(m: dict) -> None:
    verdict = ("PROFITABLE" if m["ev_per_r"] > 0.05
               else "MARGINAL" if m["ev_per_r"] > 0
               else "LOSING")
    print(f"    {m['strategy']:35s}  {m['symbol']}")
    print(f"      Trades: {m['closed']} closed ({m['per_day']:.1f}/day) | "
          f"Timeouts: {m['timeouts']}")
    print(f"      WR: {m['wr']*100:.1f}%  |  Avg R:R: 1:{m['avg_rr']:.2f}  |  "
          f"Profit Factor: {m['profit_f']:.2f}")
    print(f"      EV/trade: {m['ev_per_r']:+.3f}R  |  Net: {m['net_r']:+.1f}R  |  "
          f"Max DD: {m['max_dd_r']:.1f}R  |  Max streak L: {m['max_streak']}")
    print(f"      ▶  {verdict}")
    print()


# ═══════════════════════════════════════════════════════════════
# Strategy runners
# ═══════════════════════════════════════════════════════════════

def run_mtf(bars_5m, bars_1h, config=None) -> list[tuple]:
    cfg      = config or {"ema_period": 200, "slope_bars": 3,
                          "rsi_period": 14, "rsi_entry": 40.0, "tp_rr": 2.0}
    strat    = MTFPullbackStrategy(cfg)
    closes_1h = [b["close"] for b in bars_1h]
    from pro_bot.indicators import ema, rsi
    ema_1h = ema(closes_1h, cfg.get("ema_period", 200))

    results = []
    ratio   = GRAN_1H // GRAN_5M
    for i, bar in enumerate(bars_5m):
        j = min(i // ratio, len(bars_1h) - 1)
        strat._htf_bars = bars_1h[:j + 1]
        sig = strat.feed(bar)
        if sig.action in ("BUY", "SELL"):
            results.append((i, sig))
    return results


def run_stoch(bars_5m, config=None) -> list[tuple]:
    cfg   = config or {"k_period": 5, "d_period": 3, "ob": 80.0,
                       "os_level": 20.0, "ema_period": 50,
                       "slope_bars": 3, "tp_rr": 1.5}
    strat = StochEMAStrategy(cfg)
    return strat.evaluate(bars_5m)


def run_session(bars_5m, config=None) -> list[tuple]:
    cfg   = config or {"tp_rr": 1.5}
    strat = SessionBreakoutStrategy(cfg)
    return strat.evaluate(bars_5m)


def run_divergence(bars_5m, config=None) -> list[tuple]:
    cfg   = config or {"rsi_period": 14, "lookback": 12,
                       "os_level": 40.0, "ob_level": 60.0,
                       "hidden_divergence": True, "tp_rr": 2.0}
    strat = RSIDivergenceStrategy(cfg)
    return strat.evaluate(bars_5m)


def run_pivot(bars_5m, bars_1d, config=None) -> list[tuple]:
    cfg   = config or {"rsi_period": 14, "rsi_os": 40.0,
                       "tol_pct": 0.001, "tp_rr": 1.5}
    strat = PivotBounceStrategy(cfg)
    results = []
    day_idx = 0
    for i, bar in enumerate(bars_5m):
        epoch = bar["epoch"]
        # Advance daily bar pointer
        while day_idx + 1 < len(bars_1d) and bars_1d[day_idx + 1]["epoch"] <= epoch:
            day_idx += 1
            strat.feed_daily_bar(bars_1d[day_idx])
        sig = strat.feed(bar)
        if sig.action in ("BUY", "SELL"):
            results.append((i, sig))
    return results


# ═══════════════════════════════════════════════════════════════
# Parameter sweep per strategy
# ═══════════════════════════════════════════════════════════════

def sweep_mtf(bars_5m, bars_1h, sym) -> list[dict]:
    results = []
    for ema_p in [100, 200]:
        for rsi_e in [35, 40, 45]:
            for rr in [1.5, 2.0]:
                cfg  = {"ema_period": ema_p, "slope_bars": 3,
                        "rsi_period": 14, "rsi_entry": rsi_e, "tp_rr": rr}
                sigs = run_mtf(bars_5m, bars_1h, cfg)
                if not sigs:
                    continue
                trades = simulate_exits(bars_5m, sigs)
                m = calc_metrics(trades, sym,
                                 f"MTF EMA{ema_p} RSI<{rsi_e} RR{rr}")
                if m:
                    results.append(m)
    return results


def sweep_stoch(bars_5m, sym) -> list[dict]:
    results = []
    for kp in [5, 9]:
        for ob in [75, 80]:
            for ep in [20, 50]:
                for rr in [1.5, 2.0]:
                    cfg  = {"k_period": kp, "d_period": 3, "ob": ob,
                            "os_level": 100-ob, "ema_period": ep,
                            "slope_bars": 3, "tp_rr": rr}
                    sigs = run_stoch(bars_5m, cfg)
                    if not sigs:
                        continue
                    trades = simulate_exits(bars_5m, sigs)
                    m = calc_metrics(trades, sym,
                                     f"STOCH({kp},3) OB={ob} EMA{ep} RR{rr}")
                    if m:
                        results.append(m)
    return results


def sweep_session(bars_5m, sym) -> list[dict]:
    results = []
    for rr in [1.5, 2.0]:
        cfg    = {"tp_rr": rr}
        sigs   = run_session(bars_5m, cfg)
        if not sigs:
            continue
        trades = simulate_exits(bars_5m, sigs)
        m = calc_metrics(trades, sym, f"SessionBreakout RR{rr}")
        if m:
            results.append(m)
    return results


def sweep_div(bars_5m, sym) -> list[dict]:
    results = []
    for lb in [8, 12, 16]:
        for os_l in [35, 40, 45]:
            for hidden in [True, False]:
                for rr in [1.5, 2.0]:
                    cfg  = {"rsi_period": 14, "lookback": lb,
                            "os_level": os_l, "ob_level": 100-os_l,
                            "hidden_divergence": hidden, "tp_rr": rr}
                    sigs = run_divergence(bars_5m, cfg)
                    if not sigs:
                        continue
                    trades = simulate_exits(bars_5m, sigs)
                    tag = f"RSIDiv lb={lb} OS={os_l} {'H' if hidden else 'C'} RR{rr}"
                    m = calc_metrics(trades, sym, tag)
                    if m:
                        results.append(m)
    return results


def sweep_pivot(bars_5m, bars_1d, sym) -> list[dict]:
    results = []
    for rsi_os in [35, 40, 45]:
        for tol in [0.0005, 0.001, 0.002]:
            for rr in [1.5, 2.0]:
                cfg  = {"rsi_period": 14, "rsi_os": rsi_os,
                        "tol_pct": tol, "tp_rr": rr}
                sigs = run_pivot(bars_5m, bars_1d, cfg)
                if not sigs:
                    continue
                trades = simulate_exits(bars_5m, sigs)
                m = calc_metrics(trades, sym,
                                 f"PivotBounce OS={rsi_os} tol={tol} RR{rr}")
                if m:
                    results.append(m)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 75)
    print("PRO BOT BACKTEST — Dynamic SL/TP simulation")
    print("Breakeven: 1:1.5 R:R → WR>40% | 1:2 R:R → WR>33%")
    print("=" * 75)

    all_metrics: list[dict] = []

    for sym, info in SYMBOLS.items():
        if FILTER_SYM and sym != FILTER_SYM:
            continue

        print(f"\n{'─'*75}")
        print(f"[{sym}] {info['label']}")
        print(f"{'─'*75}")

        try:
            bars_5m, bars_1h, bars_1d = await load_data(sym)
            print(f"  Data: {len(bars_5m)} × 5min | {len(bars_1h)} × 1h | "
                  f"{len(bars_1d)} × daily")
        except Exception as e:
            print(f"  SKIP — {e}")
            continue

        strategies_to_run = []
        if not FILTER_STRAT or "mtf"     in FILTER_STRAT: strategies_to_run.append("mtf")
        if not FILTER_STRAT or "stoch"   in FILTER_STRAT: strategies_to_run.append("stoch")
        if not FILTER_STRAT or "session" in FILTER_STRAT: strategies_to_run.append("session")
        if not FILTER_STRAT or "div"     in FILTER_STRAT: strategies_to_run.append("div")
        if not FILTER_STRAT or "pivot"   in FILTER_STRAT: strategies_to_run.append("pivot")

        sym_metrics: list[dict] = []

        if "mtf" in strategies_to_run:
            print("  [MTF] sweeping...")
            sym_metrics += sweep_mtf(bars_5m, bars_1h, sym)

        if "stoch" in strategies_to_run:
            print("  [STOCH] sweeping...")
            sym_metrics += sweep_stoch(bars_5m, sym)

        if "session" in strategies_to_run:
            print("  [SESSION] sweeping...")
            sym_metrics += sweep_session(bars_5m, sym)

        if "div" in strategies_to_run:
            print("  [DIV] sweeping...")
            sym_metrics += sweep_div(bars_5m, sym)

        if "pivot" in strategies_to_run:
            print("  [PIVOT] sweeping...")
            sym_metrics += sweep_pivot(bars_5m, bars_1d, sym)

        all_metrics += sym_metrics
        profitable = sum(1 for m in sym_metrics if m["ev_per_r"] > 0.05)
        print(f"  → {profitable} profitable configs found")

    # ── Report ──────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("RESULTS — EV > 0.05R (profitable), sorted by EV per trade")
    print(f"{'='*75}\n")

    profitable = sorted(
        [m for m in all_metrics if m["ev_per_r"] > 0.05],
        key=lambda x: -x["ev_per_r"],
    )
    marginal = sorted(
        [m for m in all_metrics if 0 < m["ev_per_r"] <= 0.05],
        key=lambda x: -x["ev_per_r"],
    )[:10]
    losing = [m for m in all_metrics if m["ev_per_r"] <= 0]

    if profitable:
        print(f"── PROFITABLE ({len(profitable)} configs) ──\n")
        for m in profitable:
            print_metrics(m)
    else:
        print("── No profitable configs found ──\n")

    if marginal:
        print(f"── MARGINAL — top 10 (0 < EV ≤ 0.05R) ──\n")
        for m in marginal:
            print_metrics(m)

    print(f"Summary: {len(profitable)} profitable | "
          f"{len(marginal)} marginal | {len(losing)} losing "
          f"across {len(all_metrics)} total configs")

    if profitable:
        top = profitable[0]
        print(f"\nBest: {top['symbol']} — {top['strategy']} | "
              f"EV {top['ev_per_r']:+.3f}R | WR {top['wr']*100:.1f}% | "
              f"PF {top['profit_f']:.2f} | {top['per_day']:.1f}/day")


if __name__ == "__main__":
    asyncio.run(main())
