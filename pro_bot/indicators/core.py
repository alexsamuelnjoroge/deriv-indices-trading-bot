"""
Shared indicator library for pro_bot strategies.
All functions accept list[float] or list[dict] and return series.
Bar dicts: {open, high, low, close, epoch, volume(optional)}
"""


# ── EMA ──────────────────────────────────────────────────────────────────────

def ema(closes: list[float], period: int) -> list[float | None]:
    out = [None] * len(closes)
    if len(closes) < period:
        return out
    k   = 2 / (period + 1)
    val = sum(closes[:period]) / period
    out[period - 1] = val
    for i in range(period, len(closes)):
        val    = closes[i] * k + val * (1 - k)
        out[i] = val
    return out


# ── RSI ──────────────────────────────────────────────────────────────────────

def rsi(closes: list[float], period: int) -> list[float | None]:
    out = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    ch = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    g  = sum(c for c in ch[:period] if c > 0) / period
    l  = sum(-c for c in ch[:period] if c < 0) / period
    for i in range(period, len(closes)):
        d = ch[i - 1]
        g = (g * (period - 1) + max(d, 0)) / period
        l = (l * (period - 1) + max(-d, 0)) / period
        out[i] = 100.0 if l == 0 else round(100 - 100 / (1 + g / l), 2)
    return out


# ── Stochastic (%K, %D) ───────────────────────────────────────────────────────

def stochastic(bars: list[dict], k_period: int = 5,
               d_period: int = 3) -> tuple[list, list]:
    """Returns (smooth_%K, %D) series."""
    n      = len(bars)
    highs  = [b["high"]  for b in bars]
    lows   = [b["low"]   for b in bars]
    closes = [b["close"] for b in bars]

    raw_k = [None] * n
    for i in range(k_period - 1, n):
        hi = max(highs[i - k_period + 1: i + 1])
        lo = min(lows[i - k_period + 1:  i + 1])
        raw_k[i] = 100 * (closes[i] - lo) / (hi - lo) if hi != lo else 50.0

    smooth_k = [None] * n
    for i in range(k_period + d_period - 2, n):
        vals = [raw_k[j] for j in range(i - d_period + 1, i + 1)
                if raw_k[j] is not None]
        if len(vals) == d_period:
            smooth_k[i] = sum(vals) / d_period

    d_line = [None] * n
    for i in range(k_period + 2 * d_period - 3, n):
        vals = [smooth_k[j] for j in range(i - d_period + 1, i + 1)
                if smooth_k[j] is not None]
        if len(vals) == d_period:
            d_line[i] = sum(vals) / d_period

    return smooth_k, d_line


# ── ATR ───────────────────────────────────────────────────────────────────────

def atr(bars: list[dict], period: int = 14) -> list[float | None]:
    n      = len(bars)
    highs  = [b["high"]  for b in bars]
    lows   = [b["low"]   for b in bars]
    closes = [b["close"] for b in bars]

    tr  = [None] * n
    out = [None] * n

    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        )
    if n < period + 1:
        return out
    out[period] = sum(tr[1: period + 1]) / period
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def bollinger(closes: list[float], period: int = 20,
              n_std: float = 2.0) -> list[tuple | None]:
    """Returns list of (upper, mid, lower) or None."""
    out = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        w    = closes[i - period + 1: i + 1]
        mean = sum(w) / period
        std  = (sum((x - mean) ** 2 for x in w) / period) ** 0.5
        out[i] = (mean + n_std * std, mean, mean - n_std * std)
    return out


# ── Pivot Levels ─────────────────────────────────────────────────────────────

def pivot_levels(bar: dict) -> dict:
    """
    Classic floor trader pivots from a single prior-period OHLC bar.
    Returns {P, R1, R2, R3, S1, S2, S3}.
    """
    h, l, c = bar["high"], bar["low"], bar["close"]
    p  = (h + l + c) / 3
    r1 = 2 * p - l
    s1 = 2 * p - h
    r2 = p + (h - l)
    s2 = p - (h - l)
    r3 = h + 2 * (p - l)
    s3 = l - 2 * (h - p)
    return {"P": p, "R1": r1, "R2": r2, "R3": r3,
            "S1": s1, "S2": s2, "S3": s3}
