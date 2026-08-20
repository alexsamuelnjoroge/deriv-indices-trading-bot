"""
Multiplier (MULTUP/MULTDOWN) backtest for Crash/Boom post-spike recoil.

Simulates real multiplier contract mechanics tick-by-tick:
  - Enter MULTUP after CRASH spike, MULTDOWN after BOOM spike
  - Hold until TP hits (win), SL hits (loss), or max_hold_ticks expires
  - Commission deducted at open ($0.02 per $1 stake from API check)

Sweeps SL/TP combinations to find the optimal risk parameters.

Usage:
  python backtest_multiplier.py                          # all 5 symbols, sweep SL/TP
  python backtest_multiplier.py --symbol CRASH1000
  python backtest_multiplier.py --multiplier 200
  python backtest_multiplier.py --sl 0.001 --tp 0.003   # fixed SL/TP, no sweep
  python backtest_multiplier.py --fresh                  # re-download tick data
"""

import argparse
import sys

from loguru import logger

from src.data.history import fetch_ticks
from src.data.tick_store import TickStore
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")

# Per-symbol config: same spike detection params as ACCU backtest
CONFIGS = {
    "CRASH1000": {"symbol_type": "crash", "spike_mult": 15.0, "barrier_pct": 2.35e-6},
    "CRASH900":  {"symbol_type": "crash", "spike_mult": 15.0, "barrier_pct": 2.61e-6},
    "CRASH600":  {"symbol_type": "crash", "spike_mult": 15.0, "barrier_pct": 3.93e-6},
    "BOOM900":   {"symbol_type": "boom",  "spike_mult": 15.0, "barrier_pct": 2.62e-6},
    "BOOM600":   {"symbol_type": "boom",  "spike_mult": 15.0, "barrier_pct": 3.95e-6},
}

STAKE             = 1.0
COMMISSION        = 0.02   # flat per trade from API: $0.02 per $1 stake at 100x

# SL/TP sweep: expressed as fraction of PRICE MOVE.
# Multiply by (stake × multiplier) to get dollar amounts.
# At 100x: sl_pct=0.001 → SL=$0.10  (price moves 0.1% against us)
SL_VALUES = [0.0005, 0.001, 0.002, 0.003]   # 0.05%, 0.1%, 0.2%, 0.3%
TP_VALUES = [0.001, 0.002, 0.003, 0.005, 0.010]  # 0.1%, 0.2%, 0.3%, 0.5%, 1.0%

SEP  = "=" * 72
THIN = "-" * 72


def _be_wr(sl_dollar: float, tp_dollar: float, commission: float) -> float:
    """Win rate needed to break even: wr*(tp-c) = (1-wr)*(sl+c)."""
    return (sl_dollar + commission) / (tp_dollar + sl_dollar) * 100


def simulate_window(
    ticks: list[dict],
    symbol_type: str,
    spike_mult: float,
    barrier_pct: float,
    multiplier: int,
    sl_pct: float,
    tp_pct: float,
    max_hold: int,
) -> dict:
    """Replay one data window through the strategy with multiplier settlement."""
    strategy_cfg = {
        "symbol_type":    symbol_type,
        "spike_mult":     spike_mult,
        "atr_period":     50,
        "cooldown_ticks": 5,
        "loss_cooldown":  0,
        "barrier_pct":    barrier_pct,
        "confirm_threshold": 1.0,
        "use_binary":     True,  # returns BUY_RISE / BUY_FALL
    }
    strategy   = CrashBoomRecoilStrategy(strategy_cfg)
    tick_store = TickStore(rsi_period=14, bar_size=1)

    sl_dollar = STAKE * multiplier * sl_pct
    tp_dollar = STAKE * multiplier * tp_pct

    balance = 0.0
    trades  = []
    pending = None

    for i, raw in enumerate(ticks):
        tick_store.add(raw)
        price = float(raw["quote"])

        # ── Check open multiplier contract ─────────────────────────────
        if pending is not None:
            ep = pending["entry_price"]
            ct = pending["contract_type"]

            pnl = STAKE * multiplier * (
                (price - ep) / ep if ct == "MULTUP" else (ep - price) / ep
            )
            held = i - pending["entry_idx"]
            closed = False

            if pnl >= tp_dollar:
                profit, won = tp_dollar - COMMISSION, True
                closed = True
            elif pnl <= -sl_dollar:
                profit, won = -(sl_dollar + COMMISSION), False
                closed = True
            elif held >= max_hold:
                profit = pnl - COMMISSION
                won    = profit > 0
                closed = True

            if closed:
                balance += profit
                trades.append({"won": won, "profit": profit, "held": held})
                strategy.on_result(won)
                pending = None

        # ── Evaluate strategy for new entry ────────────────────────────
        if pending is None:
            signal = strategy.evaluate(tick_store)
            if signal.action in ("BUY_RISE", "BUY_FALL"):
                ct = "MULTUP" if signal.action == "BUY_RISE" else "MULTDOWN"
                pending = {
                    "entry_idx":    i,
                    "entry_price":  price,
                    "contract_type": ct,
                }

    total = len(trades)
    wins  = sum(1 for t in trades if t["won"])
    net   = sum(t["profit"] for t in trades)
    wr    = wins / total * 100 if total > 0 else 0.0
    avg_h = sum(t["held"] for t in trades) / total if total > 0 else 0.0

    return {
        "trades": total, "wins": wins, "wr": wr,
        "net": net, "avg_held": avg_h,
        "be_wr": _be_wr(sl_dollar, tp_dollar, COMMISSION),
    }


def run_symbol(
    symbol: str,
    cfg: dict,
    multiplier: int,
    num_windows: int,
    window_size: int,
    max_hold: int,
    sl_fixed: float | None,
    tp_fixed: float | None,
    fresh: bool,
) -> None:
    total_ticks = window_size * num_windows
    print()
    print(SEP)
    print(f"  {symbol}  |  multiplier={multiplier}x  |  stake=${STAKE:.2f}  |  commission=${COMMISSION:.2f}/trade")
    print(f"  Fetching {total_ticks:,} ticks ({num_windows} x {window_size:,})  max_hold={max_hold}t ...")
    print(THIN)

    try:
        ticks = fetch_ticks(symbol, count=total_ticks, fresh=fresh)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    if len(ticks) < window_size:
        print(f"  Only {len(ticks)} ticks — skipping.")
        return

    ticks = ticks[-num_windows * window_size:]

    sl_values = [sl_fixed] if sl_fixed is not None else SL_VALUES
    tp_values = [tp_fixed] if tp_fixed is not None else TP_VALUES

    header = (f"  {'SL':>5}  {'TP':>5}  {'R:R':>4}  {'BE%':>5}  "
              + "  ".join(f"W{w+1} WR% / net$" for w in range(num_windows))
              + "  Result")
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_passes = -1
    best_row    = ""

    for sl_pct in sl_values:
        for tp_pct in tp_values:
            if tp_pct <= sl_pct:
                continue

            sl_d = STAKE * multiplier * sl_pct
            tp_d = STAKE * multiplier * tp_pct
            rr   = tp_pct / sl_pct
            be   = _be_wr(sl_d, tp_d, COMMISSION)

            row_parts = [f"  {sl_pct*100:>4.2f}%  {tp_pct*100:>4.2f}%  {rr:>4.1f}  {be:>5.1f}%"]
            passes = 0
            for w in range(num_windows):
                seg = ticks[w * window_size: (w + 1) * window_size]
                r   = simulate_window(
                    seg, cfg["symbol_type"], cfg["spike_mult"], cfg["barrier_pct"],
                    multiplier, sl_pct, tp_pct, max_hold,
                )
                ok = r["wr"] >= r["be_wr"] and r["trades"] >= 3
                passes += ok
                flag = "P" if ok else "F"
                row_parts.append(
                    f"  {r['wr']:>5.1f}%({r['trades']:>2}t) {r['net']:>+6.2f}{flag}"
                )
            verdict = "ROBUST" if passes == num_windows else f"{passes}/{num_windows}"
            row = "  ".join(row_parts) + f"  {verdict}"
            print(row)

            if passes > best_passes:
                best_passes = passes
                best_row    = f"sl={sl_pct:.4f}  tp={tp_pct:.4f}  (R:R {rr:.1f})"

    print(THIN)
    if best_passes == num_windows:
        print(f"  BEST: {best_row}  => all windows PASS")
        print(f"  ACTION: enable {symbol} in config.yaml with strategy_type: crash_boom_recoil")
        print(f"          multiplier: {multiplier}  sl_pct: ...  tp_pct: ...  (see ROBUST row above)")
    elif best_passes >= num_windows * 3 // 4:
        print(f"  BEST: {best_row}  => {best_passes}/{num_windows} windows pass (MOSTLY OK)")
    else:
        print(f"  No robust SL/TP found — multiplier strategy has no edge on {symbol}")


def main():
    parser = argparse.ArgumentParser(description="Multiplier contract backtest")
    parser.add_argument("--symbol",     default=None,  help="Single symbol (e.g. CRASH1000)")
    parser.add_argument("--multiplier", type=int,   default=100, help="Multiplier value (default 100)")
    parser.add_argument("--windows",    type=int,   default=4,   help="Walk-forward windows (default 4)")
    parser.add_argument("--count",      type=int,   default=15000, help="Ticks per window (default 15000)")
    parser.add_argument("--max-hold",   type=int,   default=500,  dest="max_hold",
                        help="Max ticks before force-closing (default 500)")
    parser.add_argument("--sl",         type=float, default=None,
                        help="Fixed SL as fraction of price move (e.g. 0.001 = 0.1%%). Default: sweep")
    parser.add_argument("--tp",         type=float, default=None,
                        help="Fixed TP as fraction of price move (e.g. 0.003 = 0.3%%). Default: sweep")
    parser.add_argument("--fresh",      action="store_true", help="Re-download tick data")
    args = parser.parse_args()

    if args.symbol:
        if args.symbol not in CONFIGS:
            print(f"Unknown symbol '{args.symbol}'. Available: {list(CONFIGS)}")
            return
        symbols = {args.symbol: CONFIGS[args.symbol]}
    else:
        symbols = CONFIGS

    print(f"\nMultiplier Backtest  |  {args.multiplier}x  |  {args.windows} windows × {args.count:,} ticks")
    print(f"Commission ${COMMISSION:.2f} per trade  |  max_hold={args.max_hold} ticks")
    print(f"SL/TP expressed as fraction of price move  (×{args.multiplier} = dollar P/L on $1 stake)")
    print(f"Breakeven WR formula: (SL$ + commission) / (TP$ + SL$)")

    for symbol, cfg in symbols.items():
        run_symbol(
            symbol=symbol, cfg=cfg,
            multiplier=args.multiplier,
            num_windows=args.windows,
            window_size=args.count,
            max_hold=args.max_hold,
            sl_fixed=args.sl,
            tp_fixed=args.tp,
            fresh=args.fresh,
        )

    print()
    print(SEP)
    print("Done.")
    print("Rows marked ROBUST: enable in config.yaml with the matching sl_pct / tp_pct.")


if __name__ == "__main__":
    main()
