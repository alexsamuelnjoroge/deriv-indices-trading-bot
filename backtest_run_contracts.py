"""
Backtest RUNHIGH / RUNLOW contracts on Volatility indices.

RUNHIGH: next N ticks each higher than the previous → wins if all N ticks go up.
RUNLOW:  next N ticks each lower than the previous  → wins if all N ticks go down.

Payout:
  3 ticks: 7.63x → profit 663% → BE = 13.1%   (pure random = 12.5%)
  4 ticks: higher payout (to check)
  5 ticks: higher payout (to check)

Strategy options tested:
  1. BASELINE   — buy every eligible tick (measures raw autocorrelation)
  2. POST-UP    — buy RUNHIGH only after a tick that was already UP (momentum entry)
  3. POST-DOWN  — buy RUNLOW only after a tick that was already DOWN (momentum entry)
  4. POST-LARGE — buy RUNHIGH/RUNLOW after large tick move (ATR x N)

Walk-forward: 4 windows x 15k ticks per symbol.

Usage:
  python backtest_run_contracts.py                     # all symbols, 3-tick run
  python backtest_run_contracts.py --ticks 5           # 5-tick run
  python backtest_run_contracts.py --symbol R_10
  python backtest_run_contracts.py --fresh
"""

import argparse
import sys
from math import sqrt

from loguru import logger

from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100",
           "1HZ10V", "1HZ25V", "1HZ100V"]

# Payouts from API survey — approximate; vary slightly per account
PAYOUTS = {3: 6.63, 4: 14.0, 5: 30.0}   # profit %, 3t confirmed; 4t/5t estimated

SEP  = "=" * 80
THIN = "-" * 80


def be_wr(payout_profit_pct: float) -> float:
    """Break-even WR given profit% payout (e.g. 663 -> need 13.1% WR)."""
    return 100.0 / (1.0 + payout_profit_pct / 100.0)


def tick_direction(prices: list[float], i: int) -> int:
    """Return +1 if prices[i] > prices[i-1], -1 if lower, 0 if equal."""
    if i == 0 or prices[i - 1] == 0:
        return 0
    diff = prices[i] - prices[i - 1]
    return 1 if diff > 0 else (-1 if diff < 0 else 0)


def atr(prices: list[float], period: int = 20) -> float:
    if len(prices) < period + 1:
        return 0.0
    moves = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    return sum(moves[-period:]) / period


def simulate_strategy(
    ticks: list[dict],
    run_len: int,
    direction: int,      # +1 = RUNHIGH, -1 = RUNLOW
    entry_mode: str,     # "baseline", "post_momentum", "post_large"
    atr_thresh: float,   # for post_large mode: entry after move > atr_thresh x ATR
    cooldown: int = 1,   # ticks to skip after each entry
) -> dict:
    """
    Simulate a run contract strategy on a tick sequence.
    Returns: trades, wins, wr, net_profit (assuming $1 stake).
    """
    prices = [float(t["quote"]) for t in ticks]
    n      = len(prices)
    payout_profit_pct = PAYOUTS.get(run_len, 663.0)
    be     = be_wr(payout_profit_pct)
    profit_per_win  = payout_profit_pct / 100.0   # per $1 stake
    loss_per_trade  = 1.0                           # $1 lost per losing trade

    trades   = 0
    wins     = 0
    net      = 0.0
    cooldown_remaining = 0

    for i in range(50, n - run_len):   # warmup 50 ticks
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        cur_dir = tick_direction(prices, i)

        # Entry filter
        if entry_mode == "baseline":
            enter = True
        elif entry_mode == "post_momentum":
            # Buy RUNHIGH after an UP tick, RUNLOW after DOWN tick
            enter = (cur_dir == direction)
        elif entry_mode == "post_large":
            # Buy after a large tick in the run direction
            a = atr(prices[:i], 20)
            move = abs(prices[i] - prices[i - 1])
            enter = (cur_dir == direction and a > 0 and move > atr_thresh * a)
        else:
            enter = True

        if not enter:
            continue

        # Simulate the run: next run_len ticks must all go in direction
        won = True
        for k in range(1, run_len + 1):
            d = tick_direction(prices, i + k)
            if d != direction:
                won = False
                break

        trades += 1
        if won:
            wins += 1
            net  += profit_per_win
        else:
            net  -= loss_per_trade

        cooldown_remaining = cooldown

    wr = wins / trades * 100 if trades > 0 else 0.0
    return {"trades": trades, "wins": wins, "wr": wr, "net": net,
            "be": be, "edge": wr - be, "payout_pct": payout_profit_pct}


def autocorrelation(prices: list[float], lag: int = 1) -> float:
    """Compute autocorrelation of tick returns at given lag."""
    returns = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    n = len(returns)
    if n < lag + 10:
        return float("nan")
    mean = sum(returns) / n
    var  = sum((r - mean) ** 2 for r in returns) / n
    if var == 0:
        return 0.0
    cov = sum((returns[i] - mean) * (returns[i - lag] - mean)
              for i in range(lag, n)) / (n - lag)
    return cov / var


def run_symbol(symbol: str, run_len: int, num_windows: int,
               window_size: int, fresh: bool) -> None:
    total_ticks = window_size * num_windows
    payout_pct  = PAYOUTS.get(run_len, 663.0)
    be          = be_wr(payout_pct)

    print()
    print(SEP)
    print(f"  {symbol}  |  RUNHIGH/RUNLOW {run_len}-tick  |  "
          f"payout={payout_pct:.0f}%  BE={be:.1f}%  (pure random={(0.5**run_len)*100:.1f}%)")
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
    prices_all = [float(t["quote"]) for t in ticks]

    # Autocorrelation across full dataset
    ac1 = autocorrelation(prices_all, lag=1)
    ac2 = autocorrelation(prices_all, lag=2)
    print(f"  Tick return autocorrelation: lag-1={ac1:+.4f}  lag-2={ac2:+.4f}  "
          f"({'positive momentum' if ac1 > 0.01 else 'negative (mean-reversion)' if ac1 < -0.01 else 'near-zero (random)'})")
    print()

    strategies = [
        ("BASELINE",       "baseline",      0,   1),
        ("POST-UP→HIGH",   "post_momentum", 0,   1),   # RUNHIGH after up tick
        ("POST-DOWN→LOW",  "post_momentum", 0,  -1),   # RUNLOW after down tick
        ("LARGE-UP→HIGH",  "post_large",    2.0, 1),   # RUNHIGH after 2xATR up
        ("LARGE-DOWN→LOW", "post_large",    2.0,-1),   # RUNLOW after 2xATR down
    ]

    for (label, mode, atr_t, direction) in strategies:
        window_wrs  = []
        window_pass = []
        total_trades = total_wins = 0
        total_net   = 0.0

        for w in range(num_windows):
            seg  = ticks[w * window_size: (w + 1) * window_size]
            r    = simulate_strategy(seg, run_len, direction, mode, atr_t)
            window_wrs.append(r["wr"])
            passes = r["wr"] >= r["be"] and r["trades"] >= 10
            window_pass.append(passes)
            total_trades += r["trades"]
            total_wins   += r["wins"]
            total_net    += r["net"]

        wr_overall = total_wins / total_trades * 100 if total_trades > 0 else 0.0
        edge       = wr_overall - be
        n_pass     = sum(window_pass)
        verdict    = "ROBUST" if n_pass == num_windows else f"{n_pass}/{num_windows}"
        flag       = " <<" if n_pass == num_windows and edge > 1.0 else ""

        wrs_str = "  ".join(f"W{w+1}:{window_wrs[w]:.1f}%({'P' if window_pass[w] else 'F'})"
                             for w in range(num_windows))
        print(
            f"  {label:<22}  WR={wr_overall:>5.1f}%  BE={be:.1f}%  "
            f"edge={edge:>+5.2f}%  n={total_trades:>5}  net={total_net:>+7.2f}  "
            f"{verdict}{flag}"
        )
        print(f"  {'':22}  {wrs_str}")
        print()

    print(THIN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default=None)
    parser.add_argument("--ticks",   type=int, default=3,     dest="run_len")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--count",   type=int, default=15000)
    parser.add_argument("--fresh",   action="store_true")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    print(f"\nRUN contract backtest  |  run_len={args.run_len}t  |  "
          f"{args.windows} windows x {args.count:,} ticks")
    print(f"payout={PAYOUTS.get(args.run_len, '?')}%  "
          f"BE={be_wr(PAYOUTS.get(args.run_len, 663.0)):.1f}%  "
          f"pure_random={(0.5**args.run_len)*100:.1f}%")

    for symbol in symbols:
        run_symbol(symbol, args.run_len, args.windows, args.count, args.fresh)

    print()
    print(SEP)
    print("ROBUST = all windows pass.  Edge = WR - BE.")
    print("Positive autocorrelation (lag-1 > 0) makes RUNHIGH/RUNLOW more likely to win.")


if __name__ == "__main__":
    main()
