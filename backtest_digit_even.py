"""
DIGITEVEN backtest for R_50 and R_75.

Simulates buying DIGITEVEN on every tick (1-tick duration).
Settlement: next tick's last digit is even → win; odd → loss.
Payout: 95% profit on stake (BE = 51.28%).

Usage:
  python backtest_digit_even.py                        # R_50 + R_75, 4 x 10k ticks
  python backtest_digit_even.py --symbol R_50
  python backtest_digit_even.py --windows 4 --count 15000
  python backtest_digit_even.py --fresh
"""

import argparse
import sys

from loguru import logger

from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

PRECISION = {
    "R_10": 3, "R_25": 3, "R_50": 3, "R_75": 3, "R_100": 3,
    "1HZ10V": 2, "1HZ25V": 2, "1HZ100V": 2,
    "JD10": 3, "JD25": 3,
}
PAYOUT  = 0.95    # 95% profit per win
BE      = 1 / (1 + PAYOUT) * 100   # 51.28%
STAKE   = 1.0

SEP  = "=" * 72
THIN = "-" * 72

DEFAULT_SYMBOLS = ["R_50", "R_75"]
FETCH_COUNT     = 50000   # matches cached file from digit distribution analysis


def run_window(ticks: list[dict], precision: int) -> dict:
    wins = losses = 0
    net  = 0.0
    balance = 1000.0

    for i in range(len(ticks) - 1):
        settlement_price = float(ticks[i + 1]["quote"])
        last_digit = int(round(settlement_price * (10 ** precision))) % 10
        won = (last_digit % 2 == 0)

        if won:
            wins    += 1
            net     += STAKE * PAYOUT
            balance += STAKE * PAYOUT
        else:
            losses  += 1
            net     -= STAKE
            balance -= STAKE

    trades = wins + losses
    wr     = wins / trades * 100 if trades > 0 else 0.0
    ev     = (wr - BE) / 100 * PAYOUT * 100   # edge × payout as %

    return {
        "trades": trades,
        "wins":   wins,
        "losses": losses,
        "wr":     wr,
        "ev_pct": ev,
        "net":    net,
        "pass":   wr >= BE,
    }


def main():
    parser = argparse.ArgumentParser(description="DIGITEVEN backtest")
    parser.add_argument("--symbol",  default="ALL",
                        help="Symbol or ALL (default: ALL = R_50 + R_75)")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--count",   type=int, default=10000)
    parser.add_argument("--fresh",   action="store_true")
    args = parser.parse_args()

    symbols = DEFAULT_SYMBOLS if args.symbol.upper() == "ALL" else [args.symbol.upper()]

    print(f"\nDIGITEVEN Backtest  |  {args.windows} windows x {args.count:,} ticks  "
          f"|  payout={PAYOUT*100:.0f}%  BE={BE:.2f}%  stake=${STAKE:.2f}")

    for symbol in symbols:
        precision = PRECISION.get(symbol, 3)
        print(f"\n{SEP}")
        print(f"  {symbol}  |  precision={precision}dp")
        print(SEP)

        ticks = fetch_ticks(symbol, count=FETCH_COUNT, fresh=args.fresh)
        needed = args.windows * args.count + 1
        if len(ticks) < needed:
            print(f"  Only {len(ticks)} ticks available (need {needed}) — skipping.")
            continue

        ticks = ticks[-needed:]

        print(f"  {'Window':>6}  {'Trades':>6}  {'W/L':>9}  {'WR%':>6}  "
              f"{'edge%':>7}  {'EV%/trade':>9}  {'Net $':>8}  Pass")
        print(THIN)

        total_wins = total_trades = 0
        total_net  = 0.0
        passes     = 0

        for w in range(args.windows):
            seg = ticks[w * args.count: w * args.count + args.count + 1]
            r   = run_window(seg, precision)

            total_wins   += r["wins"]
            total_trades += r["trades"]
            total_net    += r["net"]
            if r["pass"]:
                passes += 1

            wl  = f"{r['wins']}/{r['losses']}"
            tag = "PASS" if r["pass"] else "FAIL"
            print(
                f"  Win{w+1:>3}  {r['trades']:>6}  {wl:>9}  {r['wr']:>5.2f}%  "
                f"{r['wr']-BE:>+6.2f}%  {r['ev_pct']:>+9.3f}%  {r['net']:>+8.2f}  {tag}"
            )

        print(THIN)
        wr_total  = total_wins / total_trades * 100 if total_trades > 0 else 0.0
        ev_total  = (wr_total - BE) / 100 * PAYOUT * 100
        verdict   = f"{passes}/{args.windows} ROBUST" if passes == args.windows else f"{passes}/{args.windows}"
        print(
            f"  {'TOTAL':>6}  {total_trades:>6}  {total_wins}/{total_trades-total_wins:>0}  "
            f"{wr_total:>5.2f}%  {wr_total-BE:>+6.2f}%  {ev_total:>+9.3f}%  "
            f"{total_net:>+8.2f}  {verdict}"
        )

        trades_per_tick = (args.count - 1) / args.count
        print(f"\n  Kelly f*:  WR={wr_total:.1f}%  payout={PAYOUT*100:.0f}%  "
              f"f*={(wr_total/100 * PAYOUT - (1 - wr_total/100)) / PAYOUT * 100:.1f}%  "
              f"half-Kelly={(wr_total/100 * PAYOUT - (1 - wr_total/100)) / PAYOUT * 50:.1f}%")

    print(f"\n{SEP}")
    print(f"  DIGITEVEN payout={PAYOUT*100:.0f}%  BE={BE:.2f}%")
    print(f"  EV%/trade = (WR - BE) x payout")
    print(f"  Settlement: next tick's last digit is even (0,2,4,6,8)")


if __name__ == "__main__":
    main()
