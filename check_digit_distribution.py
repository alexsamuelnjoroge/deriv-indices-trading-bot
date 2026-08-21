"""
Analyze last-digit distribution of Volatility and Jump index tick prices.

If any digit appears significantly more or less than 10% of the time,
DIGITDIFF (bet the last digit will NOT be X) or DIGITMATCH (bet it WILL be X)
contracts become exploitable.

Key math:
  DIGITDIFF payout ~95%: need WR > 51.3% to be profitable.
    If digit X appears 11% of the time -> DIFFER from X wins 89% -> WR=89% >> 51.3%  HUGE EDGE
    If digit X appears 9% of the time  -> DIFFER from X wins 91% -> even better
  DIGITMATCH payout ~9x-10x: need WR > ~9% to be profitable.
    If digit X appears 15% of the time -> MATCH X wins 15% >> 9%  EDGE

Usage:
  python check_digit_distribution.py              # all symbols, 10k ticks each
  python check_digit_distribution.py --count 50000  # more ticks for better stats
  python check_digit_distribution.py --fresh      # re-download
"""

import argparse
import sys
from collections import Counter
from math import sqrt

from loguru import logger
from scipy import stats as scipy_stats

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")

from src.data.history import fetch_ticks

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100",
           "1HZ10V", "1HZ25V", "1HZ100V", "JD10", "JD25"]

SEP  = "=" * 72
THIN = "-" * 72


def last_digit(price: float) -> int:
    """Extract last decimal digit from a price string representation."""
    s = f"{price:.5f}"   # e.g. "4828.36000"
    # Strip trailing zeros and get final non-zero digit
    s_clean = s.replace(".", "")
    # Take the last non-zero digit; if all zeros, digit is 0
    for ch in reversed(s_clean):
        if ch != "0":
            return int(ch)
    return 0


def analyze_digits(ticks: list[dict]) -> dict:
    """Count last-digit frequencies and compute chi-square test for uniformity."""
    counts = Counter()
    for t in ticks:
        price = float(t["quote"])
        counts[last_digit(price)] += 1

    total = sum(counts.values())
    freq  = {d: counts.get(d, 0) / total for d in range(10)}

    # Chi-square test: are digits uniformly distributed?
    observed  = [counts.get(d, 0) for d in range(10)]
    expected  = [total / 10.0] * 10
    chi2, p   = scipy_stats.chisquare(observed, f_exp=expected)

    return {
        "total":    total,
        "counts":   dict(counts),
        "freq":     freq,
        "chi2":     chi2,
        "p_value":  p,
        "uniform":  p > 0.05,
    }


def digit_edge(freq: dict) -> list[tuple]:
    """
    For each digit, compute:
      - DIFFER edge: P(win)=1-freq[d], payout ~95%, edge over BE 51.3%
      - MATCH edge:  P(win)=freq[d], payout ~9x (need payout check), edge over BE ~10%
    Returns list of (digit, differ_wr, match_wr, recommendation) sorted by differ edge desc.
    """
    results = []
    for d in range(10):
        differ_wr = (1 - freq.get(d, 0.1)) * 100
        match_wr  = freq.get(d, 0.1) * 100
        differ_edge = differ_wr - 51.3   # vs BE at 95% payout
        match_edge  = match_wr  - 10.0   # vs BE at 9x payout (rough)
        results.append((d, differ_wr, differ_edge, match_wr, match_edge))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    print(f"\nDigit Distribution Analysis  |  {args.count:,} ticks per symbol")
    print(SEP)

    summary_rows = []

    for symbol in SYMBOLS:
        try:
            ticks = fetch_ticks(symbol, count=args.count, fresh=args.fresh)
        except RuntimeError as e:
            print(f"  {symbol}: ERROR {e}")
            continue

        ticks = ticks[-args.count:]
        result = analyze_digits(ticks)
        freq   = result["freq"]
        edges  = digit_edge(freq)

        uniform_str = "UNIFORM" if result["uniform"] else f"NON-UNIFORM (p={result['p_value']:.4f})"
        print()
        print(f"  {symbol}  |  n={result['total']:,}  |  chi2={result['chi2']:.2f}  |  {uniform_str}")
        print(THIN)

        # Print digit frequency table
        print(f"  Digit  {'Freq%':>6}  {'Count':>6}  {'DIFFER WR%':>10}  {'DIFFER edge':>12}  {'MATCH WR%':>10}")
        for d, differ_wr, differ_edge, match_wr, match_edge in edges:
            flag = ""
            if differ_edge > 3:
                flag = "  *** STRONG DIFFER EDGE"
            elif differ_edge > 1:
                flag = "  *   DIFFER EDGE"
            elif match_edge > 2:
                flag = "  **  MATCH EDGE"
            print(
                f"  {d:>5}  {freq.get(d,0)*100:>5.2f}%  {result['counts'].get(d,0):>6}  "
                f"{differ_wr:>10.2f}%  {differ_edge:>+11.2f}%  {match_wr:>10.2f}%{flag}"
            )

        # Best differ opportunity
        best = edges[0]
        best_digit, best_differ_wr, best_differ_edge = best[0], best[1], best[2]
        summary_rows.append({
            "symbol": symbol, "n": result["total"],
            "uniform": result["uniform"], "p": result["p_value"],
            "best_digit": best_digit, "best_differ_wr": best_differ_wr,
            "best_differ_edge": best_differ_edge,
        })

    # Summary table
    print()
    print(SEP)
    print("  SUMMARY — Best DIFFER opportunity per symbol")
    print(THIN)
    print(f"  {'Symbol':<10}  {'n':>6}  {'Uniform?':>9}  p-value  "
          f"  Best digit  {'DIFFER WR%':>10}  {'Edge over BE':>12}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*7}  "
          f"  {'-'*10}  {'-'*10}  {'-'*12}")

    for r in summary_rows:
        uni   = "YES" if r["uniform"] else "NO *"
        flag  = "  << EDGE" if r["best_differ_edge"] > 1.5 else ""
        print(
            f"  {r['symbol']:<10}  {r['n']:>6}  {uni:>9}  {r['p']:>7.4f}"
            f"  {r['best_digit']:>11}  {r['best_differ_wr']:>10.2f}%  "
            f"{r['best_differ_edge']:>+11.2f}%{flag}"
        )

    print()
    print("DIFFER payout ~95% -> BE = 51.3%.  Edge = DIFFER WR - 51.3%.")
    print("MATCH payout ~9x  -> BE = 10%.     Edge = MATCH WR - 10%.")
    print("Non-uniform (p<0.05): digit frequencies deviate from expected 10%.")
    print("Strong edge: run --count 50000 on promising symbols to confirm.")


if __name__ == "__main__":
    main()
