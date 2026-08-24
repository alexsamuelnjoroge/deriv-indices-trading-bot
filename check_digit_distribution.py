"""
Analyze last-digit distribution of Volatility and Jump index tick prices.

FIXED: Uses precision-aware last digit extraction.
Prior version incorrectly stripped trailing zeros (Python floats lose 9382.450 → 9382.45),
causing digit 0 to appear never — a pure artifact. This version detects pip precision
from the tick data and extracts the correct last digit.

Usage:
  python check_digit_distribution.py              # all symbols, 10k ticks each
  python check_digit_distribution.py --count 50000
  python check_digit_distribution.py --fresh
"""

import argparse
import sys
from collections import Counter

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")

from src.data.history import fetch_ticks

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100",
           "1HZ10V", "1HZ25V", "1HZ100V", "JD10", "JD25"]

# Known pip sizes (decimal places) per symbol
# Used to extract the correct last digit for digit contracts
SYMBOL_PRECISION = {
    # Verified 2026-08-24 from live tick samples and 5dp zero-check:
    # dp+1 gives 100% WR (all zeros) confirming dp is the true last digit.
    "R_10":    4,   # prices e.g. 9382.4501 — 4dp
    "R_25":    4,   # prices e.g. 1234.5678 — 4dp
    "R_50":    4,   # prices e.g. 89.9166   — 4dp (was wrong: 3dp gave artifact chi2=380)
    "R_75":    4,   # prices e.g. 38767.6761 — 4dp (was wrong: 3dp gave artifact chi2=470)
    "R_100":   2,   # all-zero at 3dp confirms 2dp
    "1HZ10V":  2,
    "1HZ25V":  2,
    "1HZ100V": 2,
    "JD10":    2,   # all-zero at 3dp confirms 2dp
    "JD25":    2,
}

SEP  = "=" * 72
THIN = "-" * 72


def detect_precision(prices: list[float]) -> int:
    """
    Detect price precision (decimal places) from tick data.
    Finds the number of decimal places that gives the smallest non-zero
    remainders when rounding — i.e., the pip granularity.
    """
    sample = [p for p in prices[:500] if p > 0]
    if len(sample) < 10:
        return 3

    # Try precisions 0..5; find the one where int(price * 10^p) % 10
    # gives maximum variance (non-degenerate distribution of last digits)
    best_p = 3
    best_var = -1
    for p in range(1, 6):
        scale = 10 ** p
        digits = [int(round(pr * scale)) % 10 for pr in sample]
        counts = Counter(digits)
        n_distinct = len(counts)
        # The correct precision should give ~10 distinct digit values (not degenerate)
        if n_distinct >= 8:
            # Compute variance of digit counts (more uniform = real pip level)
            avg = len(sample) / 10.0
            var = sum((c - avg) ** 2 for c in counts.values()) / 10
            # We prefer the LOWEST precision where we still see ≥8 distinct digits
            # (adding more decimal places just shifts last digit, not more informative)
            if n_distinct > best_var:
                best_p   = p
                best_var = n_distinct
            break   # take first p that satisfies ≥8 distinct digits

    return best_p


def last_digit(price: float, precision: int) -> int:
    """Extract last digit at the given decimal precision."""
    scale = 10 ** precision
    return int(round(price * scale)) % 10


def _chi2_sig(chi2: float) -> str:
    """Significance label for chi-square with df=9 (10 digit categories)."""
    if chi2 >= 27.88:
        return "NON-UNIFORM p<0.001 ***"
    if chi2 >= 21.67:
        return "NON-UNIFORM p<0.01  **"
    if chi2 >= 16.92:
        return "NON-UNIFORM p<0.05  *"
    if chi2 >= 14.68:
        return "borderline  p<0.10  ~"
    return "UNIFORM     p>0.10"


def analyze_digits(ticks: list[dict], precision: int) -> dict:
    counts = Counter()
    for t in ticks:
        counts[last_digit(float(t["quote"]), precision)] += 1

    total = sum(counts.values())
    freq  = {d: counts.get(d, 0) / total for d in range(10)}

    expected_each = total / 10.0
    chi2 = sum((counts.get(d, 0) - expected_each) ** 2 / expected_each
               for d in range(10) if expected_each > 0)
    sig = _chi2_sig(chi2)

    return {"total": total, "counts": dict(counts), "freq": freq,
            "chi2": chi2, "sig": sig, "uniform": "UNIFORM" in sig}


def digit_edge(freq: dict) -> list[tuple]:
    """
    For each digit, compute differ and match edges.
    DIGITDIFF payout ~95% (profit) → BE = 51.3%
    DIGITMATCH: payout varies by digit (inverse of frequency)
    """
    results = []
    for d in range(10):
        differ_wr   = (1 - freq.get(d, 0.1)) * 100
        match_wr    = freq.get(d, 0.1) * 100
        differ_edge = differ_wr - 51.3
        match_edge  = match_wr  - 10.0
        results.append((d, differ_wr, differ_edge, match_wr, match_edge))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    print(f"\nDigit Distribution Analysis (precision-aware)  |  {args.count:,} ticks/symbol")
    print(SEP)

    summary_rows = []

    for symbol in SYMBOLS:
        try:
            ticks = fetch_ticks(symbol, count=args.count, fresh=args.fresh)
        except RuntimeError as e:
            print(f"  {symbol}: ERROR {e}")
            continue

        ticks  = ticks[-args.count:]
        prices = [float(t["quote"]) for t in ticks]

        # Determine precision
        precision = SYMBOL_PRECISION.get(symbol)
        if precision is None:
            precision = detect_precision(prices)
            src = "detected"
        else:
            src = "known"

        result = analyze_digits(ticks, precision)
        freq   = result["freq"]
        edges  = digit_edge(freq)

        print()
        print(f"  {symbol}  |  precision={precision}dp ({src})  |  n={result['total']:,}  |  "
              f"chi2={result['chi2']:.2f}  |  {result['sig']}")
        print(THIN)
        print(f"  {'Digit':>5}  {'Freq%':>6}  {'Count':>6}  {'DIFFER WR%':>10}  "
              f"{'DIFFER edge':>12}  {'MATCH WR%':>10}")
        for d, differ_wr, differ_edge, match_wr, match_edge in edges:
            flag = ""
            if differ_edge > 5:
                flag = "  *** STRONG DIFFER EDGE"
            elif differ_edge > 2:
                flag = "  *   DIFFER EDGE"
            elif match_edge > 2:
                flag = "  **  MATCH EDGE"
            print(
                f"  {d:>5}  {freq.get(d,0)*100:>5.2f}%  {result['counts'].get(d,0):>6}  "
                f"{differ_wr:>10.2f}%  {differ_edge:>+11.2f}%  {match_wr:>10.2f}%{flag}"
            )

        best = edges[0]
        summary_rows.append({
            "symbol": symbol, "n": result["total"], "precision": precision,
            "sig": result["sig"], "uniform": result["uniform"],
            "chi2": result["chi2"],
            "best_digit": best[0], "best_differ_wr": best[1],
            "best_differ_edge": best[2],
        })

    print()
    print(SEP)
    print("  SUMMARY — Best DIFFER opportunity per symbol")
    print(THIN)
    print(f"  {'Symbol':<10}  {'prec':>4}  {'n':>6}  {'chi2':>6}  {'Significance':<25}  "
          f"{'Best digit':>10}  {'DIFFER WR%':>10}  {'Edge over BE':>12}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*25}  "
          f"  {'-'*9}  {'-'*10}  {'-'*12}")

    for r in summary_rows:
        flag = "  << EDGE" if r["best_differ_edge"] > 5 else ""
        print(
            f"  {r['symbol']:<10}  {r['precision']:>4}  {r['n']:>6}  {r['chi2']:>6.1f}  "
            f"{r['sig']:<25}  {r['best_digit']:>10}  "
            f"{r['best_differ_wr']:>10.2f}%  {r['best_differ_edge']:>+11.2f}%{flag}"
        )

    print()
    print("DIFFER payout ~95% -> BE = 51.3%.  Edge = DIFFER WR - 51.3%.")
    print("DIGITEVEN/ODD payout ~95% -> BE = 51.3%.  Uniform dist: WR=50% -> -1.3% edge.")
    print("Non-uniform (chi2 > 16.92 at df=9 = p<0.05).")
    print("NOTE: 'DIFFER edge' here assumes DIGITDIFF payout is ~95% for all digits.")
    print("      Run check_digit_payouts.py to verify actual per-digit payouts.")


if __name__ == "__main__":
    main()
