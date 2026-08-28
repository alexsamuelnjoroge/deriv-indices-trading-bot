"""
DIGITOVER / DIGITUNDER edge sweep for Deriv volatility indices.

Digit 0 is near-absent on all R_ indices, boosting DIGITOVER(4) win rate
to ~55-56% against a 51.28% breakeven on the 95% payout.

Walk-forward: 2x30k tick windows. Minimum 2/2 required.

Payouts verified live (R_50, 2026-08):
  DIGITOVER(4)  -> 95%  payout -> BE = 51.28%
  DIGITUNDER(5) -> 95%  payout -> BE = 51.28%   (digit-0 scarcity hurts this side)
  DIGITOVER(3)  -> 63%  payout -> BE = 61.35%   (not viable at observed WR)
  DIGITUNDER(6) -> 63%  payout -> BE = 61.35%   (not viable at observed WR)

Only DIGITOVER(4) is expected to pass. Others are included for completeness.

Usage:
  python sweep_digit_ov_under.py
  python sweep_digit_ov_under.py --symbol R_50
"""

import argparse
import json
from pathlib import Path

CACHE_DIR   = Path("data")
TICK_COUNT  = 60_000
WINDOWS     = 2
WINDOW_SIZE = 30_000

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

# (contract_type, barrier, payout_pct, description)
# Payouts verified live from Deriv API on R_50, August 2026
BETS = [
    ("DIGITOVER",  4, 0.95, "digit in {5,6,7,8,9}"),
    ("DIGITUNDER", 5, 0.95, "digit in {0,1,2,3,4}"),
    ("DIGITOVER",  3, 0.63, "digit in {4,5,6,7,8,9}"),
    ("DIGITUNDER", 6, 0.63, "digit in {0,1,2,3,4,5}"),
    ("DIGITOVER",  5, 0.95, "digit in {6,7,8,9}"),   # half the digits; might vary
    ("DIGITUNDER", 4, 0.63, "digit in {0,1,2,3}"),
]


def last_digit(quote: str) -> int:
    s = str(quote).strip().replace("-", "").replace(".", "").replace(",", "")
    return int(s[-1]) if s else -1


def wins_bet(digit: int, contract_type: str, barrier: int) -> bool:
    if contract_type == "DIGITOVER":
        return digit > barrier
    elif contract_type == "DIGITUNDER":
        return digit < barrier
    return False


def run_window(ticks: list[dict], contract_type: str, barrier: int) -> tuple[int, int]:
    wins = total = 0
    for t in ticks:
        d = last_digit(str(t["quote"]))
        if d < 0:
            continue
        total += 1
        if wins_bet(d, contract_type, barrier):
            wins += 1
    return wins, total


def analyze(symbol: str, bet_filter: str | None = None) -> None:
    path = CACHE_DIR / f"{symbol}_{TICK_COUNT}.json"
    if not path.exists():
        print(f"  {symbol}: no cache at {path}")
        return

    with open(path) as f:
        ticks = json.load(f)

    SEP = "=" * 80
    print()
    print(SEP)
    print(f"  {symbol}  |  {len(ticks):,} ticks  |  2x{WINDOW_SIZE:,} walk-forward")
    print(SEP)
    print(f"  {'Bet':>18}  {'payout':>6}  {'BE%':>5}  "
          f"{'WR(w1)':>7}  {'WR(w2)':>7}  {'WR(all)':>7}  {'EV%':>7}  pass  description")
    print(f"  {'-'*18}  {'-'*6}  {'-'*5}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  ----  -----------")

    best_ev   = None
    best_line = None

    for ct, barrier, payout_pct, desc in BETS:
        if bet_filter and f"{ct}({barrier})" != bet_filter:
            continue

        be = 100 / (1 + payout_pct)

        window_results = []
        for w in range(WINDOWS):
            seg  = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
            wins, total = run_window(seg, ct, barrier)
            wr   = wins / total * 100 if total else 0
            passed = wr >= be
            window_results.append((wins, total, wr, passed))

        # Overall
        all_wins  = sum(r[0] for r in window_results)
        all_total = sum(r[1] for r in window_results)
        all_wr    = all_wins / all_total * 100 if all_total else 0
        ev        = (all_wr - be) / 100 * payout_pct * 100
        passes    = sum(1 for r in window_results if r[3])

        wr1 = window_results[0][2] if window_results else 0
        wr2 = window_results[1][2] if len(window_results) > 1 else 0

        mark = "***" if passes == 2 and ev > 0 else ("*" if passes == 1 and ev > 0 else "")
        line = (f"  {ct}({barrier}):>18  {payout_pct*100:>5.0f}%  {be:>5.1f}%  "
                f"{wr1:>6.2f}%  {wr2:>6.2f}%  {all_wr:>6.2f}%  {ev:>+7.3f}%  "
                f"{passes}/{WINDOWS} {mark}  {desc}")
        print(f"  {ct}({barrier}):>18  {payout_pct*100:>5.0f}%  {be:>5.1f}%  "
              f"{wr1:>6.2f}%  {wr2:>6.2f}%  {all_wr:>6.2f}%  {ev:>+7.3f}%  "
              f"{passes}/{WINDOWS} {mark:<3}  {desc}")

        if ev > 0 and passes == 2:
            if best_ev is None or ev > best_ev:
                best_ev   = ev
                best_line = (ct, barrier, payout_pct, be, all_wr, ev, passes)

    if best_line:
        ct2, bar2, pay2, be2, wr2, ev2, ps2 = best_line
        edge_pct = (wr2 - be2) / be2 * 100
        print()
        print(f"  Best validated bet: {ct2}({bar2})")
        print(f"    WR={wr2:.2f}%  BE={be2:.2f}%  edge={edge_pct:.1f}% above BE  EV={ev2:+.3f}%  passes={ps2}/{WINDOWS}")
        print(f"    At $1 stake: expected +${ev2/100:.4f} per trade")
    else:
        print()
        print(f"  No 2/2 validated positive-EV bet found for {symbol}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Run only this symbol (e.g. R_50)")
    parser.add_argument("--bet",    help="Run only this bet (e.g. DIGITOVER(4))")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    print("DIGITOVER / DIGITUNDER edge sweep")
    print("Payouts: OVER(4)=95%, UNDER(5)=95%, OVER(3)=63%, UNDER(6)=63%")
    print("Walk-forward: 2x30k ticks. Need 2/2 pass. *** = validated.\n")

    for sym in symbols:
        analyze(sym, bet_filter=args.bet)

    print()
    print("=" * 80)
    print("*** = 2/2 passes AND EV > 0   * = 1/2 passes AND EV > 0")
    print("BE = breakeven win rate for the payout. EV = edge per $1 stake as %.")


if __name__ == "__main__":
    main()
