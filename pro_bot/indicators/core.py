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


# ── ADX (Average Directional Index) ──────────────────────────────────────────

def adx(bars: list[dict], period: int = 14) -> list[float | None]:
    """
    Wilder's ADX. Values above 20 indicate a trending market.
    Returns a series aligned to the input bars (None until warm-up complete).
    """
    n = len(bars)
    out = [None] * n
    if n < 2 * period + 2:
        return out

    # TR, +DM, -DM; tr_list[k] ↔ bars[k+1]
    tr_list, pdm_list, ndm_list = [], [], []
    for i in range(1, n):
        h, l   = bars[i]["high"],     bars[i]["low"]
        ph, pl = bars[i-1]["high"],   bars[i-1]["low"]
        pc     = bars[i-1]["close"]
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
        up, dn = h - ph, pl - l
        pdm_list.append(up if up > dn and up > 0 else 0.0)
        ndm_list.append(dn if dn > up and dn > 0 else 0.0)

    # Initial Wilder sums (first `period` values → bars[1..period])
    atr_s = sum(tr_list[:period])
    pdm_s = sum(pdm_list[:period])
    ndm_s = sum(ndm_list[:period])

    # DX series; dx_list[k] ↔ bars[period + 1 + k]
    dx_list: list[float] = []
    for k in range(period, len(tr_list)):
        atr_s = atr_s - atr_s / period + tr_list[k]
        pdm_s = pdm_s - pdm_s / period + pdm_list[k]
        ndm_s = ndm_s - ndm_s / period + ndm_list[k]
        if atr_s == 0:
            dx_list.append(0.0)
            continue
        pdi   = 100.0 * pdm_s / atr_s
        ndi   = 100.0 * ndm_s / atr_s
        denom = pdi + ndi
        dx_list.append(100.0 * abs(pdi - ndi) / denom if denom else 0.0)

    if len(dx_list) < period:
        return out

    # Initial ADX = mean of first `period` DX values → assigned to bars[2*period]
    adx_s = sum(dx_list[:period]) / period
    if 2 * period < n:
        out[2 * period] = adx_s

    for k in range(period, len(dx_list)):
        adx_s = (adx_s * (period - 1) + dx_list[k]) / period
        bar_idx = period + 1 + k
        if bar_idx < n:
            out[bar_idx] = adx_s

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
