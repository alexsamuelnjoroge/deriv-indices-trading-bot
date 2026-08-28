"""
Digit distribution bias analysis for volatility indices.

Checks whether the last digit of the price (0–9) is uniformly distributed.
A non-uniform distribution is exploitable via DIGITOVER / DIGITUNDER / DIGITEVEN
contracts on Deriv.

Prints:
  - Frequency of each digit vs expected 10%
  - Chi-squared p-value for each symbol
  - For any biased symbol: which DIGITOVER(x) or DIGITUNDER(x) bets beat the
    typical 95% payout breakeven (51.3%) at the detected frequency.

Usage:
  python analyze_digit_bias.py
"""

import json
import math
from pathlib import Path

CACHE_DIR  = Path("data")
TICK_COUNT = 60_000

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

# Typical Deriv payout for DIGITOVER / DIGITUNDER (1-tick, $1 stake)
# Exact payouts vary by account and symbol — confirmed values:
# DIGITEVEN: ~95% (we already exploit this on R_50)
# DIGITOVER / DIGITUNDER: typically 5.9% (very low) to 900%+ depending on digit rarity
# We approximate: for a digit threshold d, P(win) if uniform = (9-d)/10 for OVER(d)
# The actual payout is set so house edge ~= 3-5%
# We just measure actual frequencies and flag anything > 51.3% (be for 95% payout)
PAYOUT_PCT   = 0.95        # typical Deriv 1-tick binary payout
BE_THRESHOLD = 100 / (1 + PAYOUT_PCT)   # 51.28%


def last_digit(quote: str) -> int:
    """Return the last non-whitespace digit character of the price string."""
    s = quote.strip().replace("-", "")
    # Remove decimal point, keep only digits, take last
    digits_only = s.replace(".", "").replace(",", "")
    return int(digits_only[-1]) if digits_only else -1


def chi_squared_p(observed: list[int]) -> float:
    """Chi-squared goodness-of-fit against uniform distribution. Returns p-value."""
    n      = sum(observed)
    k      = len(observed)
    expect = n / k
    chi2   = sum((o - expect) ** 2 / expect for o in observed)
    # Approximate p-value using chi-squared CDF (df = k-1 = 9)
    # Use regularised incomplete gamma function approximation
    df  = k - 1
    x   = chi2 / 2
    # Series expansion of regularised gamma Q(df/2, chi2/2)
    # Good enough for our purposes
    try:
        p = _chi2_p(chi2, df)
    except Exception:
        p = float("nan")
    return p


def _chi2_p(chi2: float, df: int) -> float:
    """P(X > chi2) for chi-squared distribution with df degrees of freedom."""
    # Use Wilson-Hilferty approximation for large df, direct series for small
    if chi2 <= 0:
        return 1.0
    # Regularised upper incomplete gamma Q(a, x) via series
    a = df / 2
    x = chi2 / 2
    return _upper_gamma_reg(a, x)


def _upper_gamma_reg(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) ~ 1 - P(a, x)."""
    if x < a + 1:
        # Use series for P(a, x)
        p = _lower_gamma_series(a, x)
        return 1.0 - p
    else:
        # Use continued fraction for Q(a, x)
        return _upper_gamma_cf(a, x)


def _lower_gamma_series(a: float, x: float, max_iter: int = 200) -> float:
    if x <= 0:
        return 0.0
    term = math.exp(-x + a * math.log(x) - _log_gamma(a))
    s = ap = 1.0 / a
    for n in range(1, max_iter):
        ap *= x / (a + n)
        s  += ap
        if abs(ap) < abs(s) * 1e-10:
            break
    return s * term


def _upper_gamma_cf(a: float, x: float, max_iter: int = 200) -> float:
    if x <= 0:
        return 1.0
    fpmin = 1e-300
    b = x + 1 - a
    c = 1 / fpmin
    d = 1 / b
    h = d
    for i in range(1, max_iter):
        an = -i * (i - a)
        b += 2
        d  = an * d + b
        c  = b + an / c
        d  = 1 / max(abs(d), fpmin) * (1 if d >= 0 else -1)
        c  = max(abs(c), fpmin) * (1 if c >= 0 else -1)
        de = d * c
        h *= de
        if abs(de - 1) < 1e-10:
            break
    return math.exp(-x + a * math.log(x) - _log_gamma(a)) * h


def _log_gamma(z: float) -> float:
    """Lanczos approximation of log(Gamma(z))."""
    g  = 7
    c  = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
          771.32342877765313, -176.61502916214059, 12.507343278686905,
          -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    z -= 1
    x  = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)


def analyze(symbol: str) -> None:
    path = CACHE_DIR / f"{symbol}_{TICK_COUNT}.json"
    if not path.exists():
        print(f"  {symbol}: no cached data at {path}")
        return

    with open(path) as f:
        ticks = json.load(f)

    counts = [0] * 10
    skipped = 0
    for t in ticks:
        d = last_digit(str(t["quote"]))
        if 0 <= d <= 9:
            counts[d] += 1
        else:
            skipped += 1

    n    = sum(counts)
    exp  = n / 10
    p    = chi_squared_p(counts)

    SEP = "=" * 65
    print()
    print(SEP)
    print(f"  {symbol}  |  {n} ticks  |  chi-sq p={p:.4f}  ", end="")
    if p < 0.001:
        print("*** STRONG BIAS ***")
    elif p < 0.05:
        print("* BIAS DETECTED *")
    else:
        print("(no significant bias)")
    print(SEP)

    print(f"  {'Digit':>5}  {'Count':>7}  {'Freq%':>6}  {'vs 10%':>7}  exploitable?")
    print(f"  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*7}  -----------")

    for d, cnt in enumerate(counts):
        freq = cnt / n * 100
        diff = freq - 10.0
        # DIGITOVER(d-1) wins when last digit >= d → freq = sum(counts[d:]) / n
        # DIGITUNDER(d) wins when last digit <= d-1 → freq = sum(counts[:d]) / n
        over_freq  = sum(counts[d:]) / n * 100 if d < 10 else 0    # OVER(d-1)
        under_freq = sum(counts[:d]) / n * 100 if d > 0  else 0    # UNDER(d)

        flags = []
        if over_freq > BE_THRESHOLD:
            flags.append(f"OVER({d-1})={over_freq:.1f}%>BE")
        if under_freq > BE_THRESHOLD:
            flags.append(f"UNDER({d})={under_freq:.1f}%>BE")
        if d % 2 == 0 and sum(counts[i] for i in range(0,10,2)) / n * 100 > BE_THRESHOLD:
            pass  # we already know EVEN works

        print(f"  {d:>5}  {cnt:>7}  {freq:>5.2f}%  {diff:>+6.2f}%  {' | '.join(flags)}")

    # Summarise DIGITEVEN
    even_cnt  = sum(counts[i] for i in range(0, 10, 2))
    odd_cnt   = sum(counts[i] for i in range(1, 10, 2))
    even_freq = even_cnt / n * 100
    print()
    print(f"  EVEN digits (0,2,4,6,8): {even_freq:.2f}%  {'>>> EXPLOITABLE' if even_freq > BE_THRESHOLD else ''}")
    print(f"  ODD  digits (1,3,5,7,9): {odd_cnt/n*100:.2f}%")

    # DIGITOVER/DIGITUNDER summary
    print()
    print("  DIGITOVER(x) = last digit > x   |   DIGITUNDER(x) = last digit < x")
    print(f"  BE for 95% payout = {BE_THRESHOLD:.2f}%")
    print(f"  {'Bet':>14}  {'P(win)%':>8}  exploitable?")
    print(f"  {'-'*14}  {'-'*8}  -----------")
    for threshold in range(9):
        over_win = sum(counts[threshold+1:]) / n * 100
        if abs(over_win - 50) > 1.0 or over_win > BE_THRESHOLD:
            mark = "YES" if over_win > BE_THRESHOLD else ""
            print(f"  DIGITOVER({threshold:>1})     {over_win:>7.2f}%  {mark}")
    for threshold in range(1, 10):
        under_win = sum(counts[:threshold]) / n * 100
        if abs(under_win - 50) > 1.0 or under_win > BE_THRESHOLD:
            mark = "YES" if under_win > BE_THRESHOLD else ""
            print(f"  DIGITUNDER({threshold:>1})    {under_win:>7.2f}%  {mark}")


if __name__ == "__main__":
    print("Digit distribution analysis on 60k tick cache")
    print("Breakeven for 95% payout = 51.28%\n")
    for sym in SYMBOLS:
        analyze(sym)
    print()
