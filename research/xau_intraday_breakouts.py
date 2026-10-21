"""XAUUSD intraday / daily breakout research (out-of-sample discipline).

Families (fixed grids, no tuning):
  1. Opening-range breakout (London 08:00 Europe/London, NY 09:30 America/New_York),
     R in {15,30,60} min, k in {1,2}, FOLLOW and FADE.
  2. Volatility-compression breakout on M15, N in {8,16}.
  3. NR4 (and inside-day) on 00:00-UTC daily bars, OCO stop entries next day.

Costs: 0.30 USD/oz round trip on every trade. Conservative fills (stop wins a
same-bar stop/target tie). Trades with stop distance < 3 x cost are skipped.
Periods: DEV 2018-2021, VAL 2022-01-01..2024-07-31, RECENT 2024-08-01..end.

Data: Dukascopy M5 BID, data/duka/frxXAUUSD_M5_<YEAR>.csv. Research only.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.30
MIN_RISK = 3 * COST
PERIODS = [("DEV", "2018-01-01", "2022-01-01"),
           ("VAL", "2022-01-01", "2024-08-01"),
           ("RECENT", "2024-08-01", "2100-01-01")]


def load():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "duka", "frxXAUUSD_M5_*.csv")))
    df = pd.concat([pd.read_csv(f, index_col=0) for f in files])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df[["open", "high", "low", "close"]].astype(float).dropna()
    return df


def record(trades, t, entry, risk, pnl):
    """pnl is gross price move in trade direction."""
    net = pnl - COST
    trades.append((t, net / risk, net / entry * 1e4))


def simulate_path(direction, entry, stop, target, highs, lows, closes):
    """Walk bars after entry. Returns gross pnl in trade direction."""
    for h, l in zip(highs, lows):
        if direction > 0:
            hit_s, hit_t = l <= stop, h >= target
        else:
            hit_s, hit_t = h >= stop, l <= target
        if hit_s:
            return (stop - entry) * direction
        if hit_t:
            return (target - entry) * direction
    if len(closes) == 0:
        return 0.0
    return (closes[-1] - entry) * direction


# ---------------------------------------------------------------- ORB
def orb(df, tz, open_hm, end_hm):
    loc = df.index.tz_convert(tz)
    mins = (loc.hour * 60 + loc.minute).values
    dates = loc.normalize().tz_localize(None).values.astype("datetime64[D]")
    o_min = open_hm[0] * 60 + open_hm[1]
    e_min = end_hm[0] * 60 + end_hm[1]
    H, L, C = df.high.values, df.low.values, df.close.values
    ud, starts = np.unique(dates, return_index=True)
    ends = np.append(starts[1:], len(dates))
    results = {}
    for R in (15, 30, 60):
        for k in (1, 2):
            for mode in ("follow", "fade"):
                results[(R, k, mode)] = []
    for d, s, e in zip(ud, starts, ends):
        m = mins[s:e]
        for R in (15, 30, 60):
            rmask = (m >= o_min) & (m < o_min + R)
            if rmask.sum() < R // 5:  # need complete range
                continue
            hi = H[s:e][rmask].max()
            lo = L[s:e][rmask].min()
            rng = hi - lo
            bmask = (m >= o_min + R) & (m < o_min + R + 180) & (m < e_min)
            bidx = np.nonzero(bmask)[0]
            cb = C[s:e][bidx]
            brk = np.nonzero((cb > hi) | (cb < lo))[0]
            if len(brk) == 0:
                continue
            j = bidx[brk[0]]
            entry = C[s + j]
            up = entry > hi
            after = np.nonzero((np.arange(e - s) > j) & (m < e_min))[0] + s
            hh, ll, cc = H[after], L[after], C[after]
            t = df.index[s + j]
            for k in (1, 2):
                # FOLLOW: stop at opposite side of range, target k*range from entry
                dirn = 1 if up else -1
                stop = lo if up else hi
                risk = abs(entry - stop)
                if risk >= MIN_RISK:
                    tgt = entry + dirn * k * rng
                    pnl = simulate_path(dirn, entry, stop, tgt, hh, ll, cc)
                    record(results[(R, k, "follow")], t, entry, risk, pnl)
                # FADE: opposite direction; stop one range beyond entry,
                # target k*range back toward/through the range
                dirn = -1 if up else 1
                risk = rng
                if risk >= MIN_RISK:
                    stop = entry - dirn * rng
                    tgt = entry + dirn * k * rng
                    pnl = simulate_path(dirn, entry, stop, tgt, hh, ll, cc)
                    record(results[(R, k, "fade")], t, entry, risk, pnl)
    return results


# ---------------------------------------------------------------- compression
def compression(df):
    m15 = df.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    H, L, C = m15.high.values, m15.low.values, m15.close.values
    out = {}
    for N in (8, 16):
        hiN = m15.high.rolling(N).max()
        loN = m15.low.rolling(N).min()
        rngN = hiN - loN
        pct = rngN.rolling(250).rank(pct=True)  # rank of current within trailing 250
        sig = (pct <= 0.20).values
        hiN, loN = hiN.values, loN.values
        trades = []
        i = 250 + N
        n = len(m15)
        while i < n - 1:
            if not sig[i]:
                i += 1
                continue
            bh, bl = hiN[i], loN[i]
            box = bh - bl
            # wait up to N bars for a close beyond the (fixed) box
            entered = False
            for j in range(i + 1, min(i + 1 + N, n)):
                if C[j] > bh or C[j] < bl:
                    dirn = 1 if C[j] > bh else -1
                    entry = C[j]
                    stop = bl if dirn > 0 else bh
                    risk = abs(entry - stop)
                    exit_j = min(j + 24, n - 1)
                    if risk >= MIN_RISK:
                        tgt = entry + dirn * 2 * box
                        sl = slice(j + 1, exit_j + 1)
                        pnl = simulate_path(dirn, entry, stop, tgt, H[sl], L[sl], C[sl])
                        record(trades, m15.index[j], entry, risk, pnl)
                    i = exit_j + 1  # one position at a time (conservative: full hold)
                    entered = True
                    break
            if not entered:
                i += 1
        out[N] = trades
    return out


# ---------------------------------------------------------------- NR4 / inside day
def daily_setups(df):
    day = df.index.floor("D")
    g = df.groupby(day)
    cnt = g.size()
    valid = cnt[cnt >= 120].index  # drop Sunday stubs / holiday half-days
    d = g.agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).loc[valid]
    d["rng"] = d.high - d.low
    nr4 = (d.rng <= d.rng.rolling(4).min()).values
    inside = ((d.high < d.high.shift(1)) & (d.low > d.low.shift(1))).values
    groups = {k: v for k, v in zip(valid, [g.get_group(x) for x in valid])}
    res = {"NR4": [], "InsideDay": []}
    days = list(valid)
    for name, flag in (("NR4", nr4), ("InsideDay", inside)):
        for i in range(4, len(days) - 1):
            if not flag[i]:
                continue
            hi, lo, rng = d.high.iloc[i], d.low.iloc[i], d.rng.iloc[i]
            if rng < MIN_RISK:
                continue
            nxt = groups[days[i + 1]]
            Hn, Ln, Cn, On = nxt.high.values, nxt.low.values, nxt.close.values, nxt.open.values
            for j in range(len(nxt)):
                up, dn = Hn[j] >= hi, Ln[j] <= lo
                if not (up or dn):
                    continue
                if up and dn:  # both triggered in one bar: assume worst (stopped)
                    entry = max(On[j], hi) if On[j] >= hi else hi
                    record(res[name], nxt.index[j], entry, rng, -rng)
                    break
                dirn = 1 if up else -1
                entry = max(On[j], hi) if up else min(On[j], lo)  # gap fills at open
                stop = lo if up else hi
                risk = abs(entry - stop)
                tgt = entry + dirn * 1.5 * rng
                # entry bar: stop can be hit after entry (conservative), target ignored
                if (dirn > 0 and Ln[j] <= stop) or (dirn < 0 and Hn[j] >= stop):
                    pnl = (stop - entry) * dirn
                else:
                    pnl = simulate_path(dirn, entry, stop, tgt, Hn[j + 1:], Ln[j + 1:], Cn[j + 1:])
                    if j + 1 >= len(nxt):
                        pnl = (Cn[j] - entry) * dirn
                record(res[name], nxt.index[j], entry, risk, pnl)
                break
    return res


# ---------------------------------------------------------------- stats
def stats(trades):
    if not trades:
        return {}
    t = pd.DataFrame(trades, columns=["time", "R", "bp"]).set_index("time")
    out = {}
    for name, a, b in PERIODS:
        x = t[(t.index >= pd.Timestamp(a, tz="UTC")) & (t.index < pd.Timestamp(b, tz="UTC"))]
        n = len(x)
        if n < 2:
            out[name] = (n, np.nan, np.nan, np.nan, np.nan)
            continue
        r = x.R.values
        tstat = r.mean() / r.std(ddof=1) * np.sqrt(n)
        out[name] = (n, (r > 0).mean(), r.mean(), x.bp.mean(), tstat)
    return out


def main():
    df = load()
    print(f"Data: {df.index[0]} -> {df.index[-1]}  bars={len(df)}")
    variants = {}
    for sess, tz, o, e in (("LDN", "Europe/London", (8, 0), (16, 30)),
                           ("NY", "America/New_York", (9, 30), (16, 0))):
        for (R, k, mode), tr in orb(df, tz, o, e).items():
            variants[f"ORB_{sess}_R{R}_k{k}_{mode}"] = tr
    for N, tr in compression(df).items():
        variants[f"COMP_N{N}"] = tr
    for name, tr in daily_setups(df).items():
        variants[name] = tr

    rows = []
    for name, tr in variants.items():
        s = stats(tr)
        row = {"variant": name}
        for p, _, _ in PERIODS:
            n, wr, ar, bp, ts = s.get(p, (0, np.nan, np.nan, np.nan, np.nan))
            row.update({f"{p}_n": n, f"{p}_wr": wr, f"{p}_R": ar, f"{p}_bp": bp, f"{p}_t": ts})
        rows.append(row)
    res = pd.DataFrame(rows).set_index("variant")
    res["cand"] = (res.DEV_R > 0) & (res.DEV_t >= 2.5) & (res.VAL_R > 0) & (res.VAL_t >= 2.5)
    res["devval_t"] = res[["DEV_t", "VAL_t"]].min(axis=1)
    res = res.sort_values("devval_t", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    fmt = res.copy()
    for c in fmt.columns:
        if c.endswith("_wr"):
            fmt[c] = (fmt[c] * 100).round(0)
        elif c.endswith(("_R", "_t", "_bp")) or c == "devval_t":
            fmt[c] = fmt[c].round(2)
    print(fmt.to_string())
    nv = len(res)
    p_one = 0.0062  # one-sided P(t>=2.5) under null, per period
    print(f"\nVariants tested: {nv}. Expected chance passes if independent: "
          f"{nv * p_one ** 2:.3f} (both periods); single-period t>=2.5 by chance: {nv * p_one:.2f}")
    print("CANDIDATES:", list(res.index[res.cand]) or "none")


if __name__ == "__main__":
    main()
