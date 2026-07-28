"""
EUR/USD Strategy Research — deep sweep with forex-appropriate strategies.

Strategies tested:
  1. EMA+RSI Pullback       — gold_trend logic applied to EUR/USD
  2. RSI Mean Reversion     — pure RSI exit from oversold/overbought
  3. Stochastic Oscillator  — %K/%D cross from extreme zones
  4. Session Breakout       — N-bar range break at London / NY open
  5. MACD Momentum          — histogram zero-cross with trend confirm
  6. Donchian Breakout      — N-bar high/low break

SL/TP sweep includes tighter values appropriate for EUR/USD x100 multiplier.
All indicators are pre-computed as series (fast — no per-bar re-scanning).
"""

import json
from datetime import datetime, timezone

MULTIPLIER     = 100
COMMISSION_PCT = 0.02
GRANULARITY    = 3600
SYMBOL         = "frxEURUSD"

SL_VALUES     = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.010]
TP_VALUES     = [0.002, 0.003, 0.005, 0.0075, 0.010, 0.015, 0.020]
MAX_BARS_LIST = [12, 24, 48, 96]

# ── Load candles ──────────────────────────────────────────────────────────────
def load_candles():
    with open(f"cache_{SYMBOL}_{GRANULARITY}s_candles.json") as f:
        return json.load(f)

# ── Indicator series (pre-computed, O(n) each) ────────────────────────────────
def ema_series(values, period):
    alpha = 2 / (period + 1)
    out = [None] * len(values)
    val = None
    warm = 0
    for i, v in enumerate(values):
        if v is None: continue
        if val is None: val = v; warm = i
        else: val = alpha * v + (1 - alpha) * val
        out[i] = val
    for i in range(warm, min(warm + period - 1, len(out))):
        out[i] = None
    return out


def rsi_series(closes, period):
    """Returns RSI value for each bar (None until warmed up)."""
    out = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains = [0.0] * (period + 1)
    losses = [0.0] * (period + 1)
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0: gains[i] = d
        else:     losses[i] = -d
    avg_g = sum(gains[1:]) / period
    avg_l = sum(losses[1:]) / period
    for i in range(period, len(closes)):
        if i > period:
            d = closes[i] - closes[i - 1]
            g = d if d > 0 else 0
            l = -d if d < 0 else 0
            avg_g = (avg_g * (period - 1) + g) / period
            avg_l = (avg_l * (period - 1) + l) / period
        out[i] = 100.0 if avg_l == 0 else round(100 - 100 / (1 + avg_g / avg_l), 2)
    return out

# ── Simulation engine ─────────────────────────────────────────────────────────
def ev_stats(candles, signals, sl_pct, tp_pct, max_bars, min_n=8):
    last_exit = -1
    wins = losses = timeouts = 0
    total_ev = 0.0
    for bar_i, direction in signals:
        if bar_i <= last_exit: continue
        entry = bar_i + 1
        if entry >= len(candles): continue
        ep   = candles[entry]["open"]
        tp_p = ep * (1 + direction * tp_pct)
        sl_p = ep * (1 - direction * sl_pct)
        resolved = False
        for i in range(entry, min(entry + max_bars, len(candles))):
            h, l = candles[i]["high"], candles[i]["low"]
            if direction == 1:
                if l <= sl_p: losses += 1; total_ev -= MULTIPLIER*sl_pct+COMMISSION_PCT; resolved=True; last_exit=i; break
                if h >= tp_p: wins   += 1; total_ev += MULTIPLIER*tp_pct-COMMISSION_PCT; resolved=True; last_exit=i; break
            else:
                if h >= sl_p: losses += 1; total_ev -= MULTIPLIER*sl_pct+COMMISSION_PCT; resolved=True; last_exit=i; break
                if l <= tp_p: wins   += 1; total_ev += MULTIPLIER*tp_pct-COMMISSION_PCT; resolved=True; last_exit=i; break
        if not resolved:
            ci = min(entry + max_bars - 1, len(candles) - 1)
            pct = (candles[ci]["close"] - ep) / ep * direction
            timeouts += 1; total_ev += MULTIPLIER*pct - COMMISSION_PCT; last_exit = ci
    n = wins + losses + timeouts
    if n < min_n: return None
    wr = wins / (wins + losses) * 100 if (wins + losses) else 0
    be = (MULTIPLIER*sl_pct+COMMISSION_PCT) / (
         MULTIPLIER*tp_pct-COMMISSION_PCT + MULTIPLIER*sl_pct+COMMISSION_PCT) * 100
    return {"n": n, "wins": wins, "losses": losses, "timeouts": timeouts,
            "wr": wr, "be_wr": be, "ev": total_ev / n}


def walk_forward(candles, signal_fn, sl, tp, mb, n_splits=3, min_n=5):
    sz = len(candles) // n_splits
    evs, passes, rows = [], 0, []
    for fold in range(n_splits):
        s = fold * sz
        e = (fold+1)*sz if fold < n_splits-1 else len(candles)
        chunk = candles[s:e]
        sigs  = signal_fn(chunk)
        r     = ev_stats(chunk, sigs, sl, tp, mb, min_n=min_n)
        t0    = datetime.fromtimestamp(chunk[0]["epoch"],  tz=timezone.utc).strftime("%m/%d")
        t1    = datetime.fromtimestamp(chunk[-1]["epoch"], tz=timezone.utc).strftime("%m/%d")
        rows.append((fold+1, t0, t1, r))
        if r and r["ev"] > 0: passes += 1; evs.append(r["ev"])
    mean_ev = sum(evs)/len(evs) if evs else None
    return rows, passes, mean_ev


def print_wf(rows, passes, mean_ev):
    print(f"  Walk-forward:")
    print(f"  {'Fold':>4}  {'Period':13}  {'N':>4}  {'WR':>6}  {'EV/tr':>8}  Result")
    print(f"  {'-'*55}")
    for fold, t0, t1, r in rows:
        if r is None: print(f"  {fold:>4}  {t0}-{t1:11}  too few signals")
        else:
            print(f"  {fold:>4}  {t0}-{t1:11}  {r['n']:>4}  {r['wr']:>5.1f}%  {r['ev']:>+8.4f}  {'PASS' if r['ev']>0 else 'FAIL'}")
    if mean_ev: print(f"  Mean EV={mean_ev:+.4f}  {passes}/3 folds passing")


def verdict(passes, mean_ev):
    if passes == 3 and mean_ev and mean_ev > 0.05: return "STRONG"
    if passes == 3: return "PASS"
    if passes >= 2: return "WEAK"
    return "FAIL"


# ── Strategy 1: EMA+RSI Pullback ─────────────────────────────────────────────
def ema_rsi_signals(candles, ema_period, slope_bars, rsi_period, rsi_entry):
    closes  = [c["close"] for c in candles]
    ema     = ema_series(closes, ema_period)
    rsi     = rsi_series(closes, rsi_period)
    out = []
    for i in range(slope_bars, len(candles)):
        if ema[i] is None or ema[i-slope_bars] is None or rsi[i] is None: continue
        slope_up   = ema[i] > ema[i - slope_bars]
        slope_down = ema[i] < ema[i - slope_bars]
        if slope_up   and rsi[i] < rsi_entry:        out.append((i, +1))
        elif slope_down and rsi[i] > 100 - rsi_entry: out.append((i, -1))
    return out


def run_ema_rsi(candles):
    print("\n  [EMA+RSI Pullback]")
    best = []
    for ema_p in [50, 100, 200]:
        for slope_b in [3, 5, 10]:
            for rsi_p in [7, 14, 21]:
                for rsi_e in [35.0, 40.0, 45.0, 50.0]:
                    # pre-compute signals once per indicator combo
                    sigs_all = ema_rsi_signals(candles, ema_p, slope_b, rsi_p, rsi_e)
                    for sl in SL_VALUES:
                        for tp in TP_VALUES:
                            if tp <= sl: continue
                            for mb in MAX_BARS_LIST:
                                r = ev_stats(candles, sigs_all, sl, tp, mb)
                                if r and r["ev"] > 0:
                                    best.append({**r, "ema_p": ema_p, "sb": slope_b,
                                                 "rsi_p": rsi_p, "rsi_e": rsi_e,
                                                 "sl": sl, "tp": tp, "mb": mb})
    if not best: print("  No profitable configurations found."); return None
    best.sort(key=lambda x: x["ev"], reverse=True)
    b = best[0]
    print(f"  Best: EMA({b['ema_p']}) slope{b['sb']} RSI({b['rsi_p']})<{b['rsi_e']} "
          f"SL{b['sl']*100:.2f}%/TP{b['tp']*100:.2f}% max{b['mb']}h")
    print(f"        N={b['n']}  WR={b['wr']:.1f}%  BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")
    rows, passes, mean_ev = walk_forward(
        candles, lambda c, b=b: ema_rsi_signals(c, b["ema_p"], b["sb"], b["rsi_p"], b["rsi_e"]),
        b["sl"], b["tp"], b["mb"])
    print_wf(rows, passes, mean_ev)
    return verdict(passes, mean_ev), passes, mean_ev, b


# ── Strategy 2: RSI Mean Reversion ───────────────────────────────────────────
def rsi_reversion_signals(candles, rsi_period, oversold, overbought):
    closes = [c["close"] for c in candles]
    rsi    = rsi_series(closes, rsi_period)
    out = []
    for i in range(1, len(candles)):
        if rsi[i] is None or rsi[i-1] is None: continue
        if rsi[i-1] < oversold  and rsi[i] >= oversold:   out.append((i, +1))
        elif rsi[i-1] > overbought and rsi[i] <= overbought: out.append((i, -1))
    return out


def run_rsi_reversion(candles):
    print("\n  [RSI Mean Reversion]")
    best = []
    for rsi_p in [7, 14, 21]:
        for os_lvl, ob_lvl in [(20, 80), (25, 75), (30, 70)]:
            sigs_all = rsi_reversion_signals(candles, rsi_p, os_lvl, ob_lvl)
            for sl in SL_VALUES:
                for tp in TP_VALUES:
                    if tp <= sl: continue
                    for mb in MAX_BARS_LIST:
                        r = ev_stats(candles, sigs_all, sl, tp, mb)
                        if r and r["ev"] > 0:
                            best.append({**r, "rsi_p": rsi_p, "os": os_lvl, "ob": ob_lvl,
                                         "sl": sl, "tp": tp, "mb": mb})
    if not best: print("  No profitable configurations found."); return None
    best.sort(key=lambda x: x["ev"], reverse=True)
    b = best[0]
    print(f"  Best: RSI({b['rsi_p']}) OS={b['os']}/OB={b['ob']} "
          f"SL{b['sl']*100:.2f}%/TP{b['tp']*100:.2f}% max{b['mb']}h")
    print(f"        N={b['n']}  WR={b['wr']:.1f}%  BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")
    rows, passes, mean_ev = walk_forward(
        candles, lambda c, b=b: rsi_reversion_signals(c, b["rsi_p"], b["os"], b["ob"]),
        b["sl"], b["tp"], b["mb"])
    print_wf(rows, passes, mean_ev)
    return verdict(passes, mean_ev), passes, mean_ev, b


# ── Strategy 3: Stochastic Oscillator ────────────────────────────────────────
def stoch_signals(candles, k_period, d_period, oversold, overbought):
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    closes = [c["close"] for c in candles]
    k_vals = []
    for i in range(k_period - 1, len(candles)):
        hh = max(highs [i-k_period+1:i+1])
        ll = min(lows  [i-k_period+1:i+1])
        k_vals.append((closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50.0)
    d_vals = [sum(k_vals[i-d_period+1:i+1])/d_period for i in range(d_period-1, len(k_vals))]
    offset = k_period - 1 + d_period - 1
    out = []
    for i in range(1, len(d_vals)):
        k_now  = k_vals[i + d_period - 1]
        k_prev = k_vals[i + d_period - 2]
        d_now, d_prev = d_vals[i], d_vals[i-1]
        bar_i = i + offset
        if k_now > d_now and k_prev <= d_prev and d_now < oversold:    out.append((bar_i, +1))
        elif k_now < d_now and k_prev >= d_prev and d_now > overbought: out.append((bar_i, -1))
    return out


def run_stochastic(candles):
    print("\n  [Stochastic Oscillator]")
    best = []
    for k_p in [9, 14, 21]:
        for d_p in [3, 5]:
            for os, ob in [(20, 80), (25, 75), (30, 70)]:
                sigs_all = stoch_signals(candles, k_p, d_p, os, ob)
                for sl in SL_VALUES:
                    for tp in TP_VALUES:
                        if tp <= sl: continue
                        for mb in MAX_BARS_LIST:
                            r = ev_stats(candles, sigs_all, sl, tp, mb)
                            if r and r["ev"] > 0:
                                best.append({**r, "k": k_p, "d": d_p, "os": os, "ob": ob,
                                             "sl": sl, "tp": tp, "mb": mb})
    if not best: print("  No profitable configurations found."); return None
    best.sort(key=lambda x: x["ev"], reverse=True)
    b = best[0]
    print(f"  Best: Stoch(%K={b['k']},%D={b['d']}) OS={b['os']}/OB={b['ob']} "
          f"SL{b['sl']*100:.2f}%/TP{b['tp']*100:.2f}% max{b['mb']}h")
    print(f"        N={b['n']}  WR={b['wr']:.1f}%  BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")
    rows, passes, mean_ev = walk_forward(
        candles, lambda c, b=b: stoch_signals(c, b["k"], b["d"], b["os"], b["ob"]),
        b["sl"], b["tp"], b["mb"])
    print_wf(rows, passes, mean_ev)
    return verdict(passes, mean_ev), passes, mean_ev, b


# ── Strategy 4: Session Breakout ─────────────────────────────────────────────
LONDON = {7, 8, 9}
NY     = {13, 14, 15}

def session_breakout_signals(candles, lookback, session_hours):
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    out = []
    for i in range(lookback, len(candles)):
        hour = datetime.fromtimestamp(candles[i]["epoch"], tz=timezone.utc).hour
        if hour not in session_hours: continue
        ch_high = max(highs[i-lookback:i])
        ch_low  = min(lows [i-lookback:i])
        if candles[i]["high"] > ch_high: out.append((i, +1))
        elif candles[i]["low"] < ch_low:  out.append((i, -1))
    return out


def run_session_breakout(candles):
    print("\n  [Session Breakout (London + NY open)]")
    best = []
    for lookback in [4, 8, 12, 24]:
        for sess_name, sess_hours in [("London", LONDON), ("NY", NY), ("Both", LONDON | NY)]:
            sigs_all = session_breakout_signals(candles, lookback, sess_hours)
            for sl in SL_VALUES:
                for tp in TP_VALUES:
                    if tp <= sl: continue
                    for mb in MAX_BARS_LIST:
                        r = ev_stats(candles, sigs_all, sl, tp, mb)
                        if r and r["ev"] > 0:
                            best.append({**r, "lb": lookback, "sess": sess_name,
                                         "sh": sess_hours, "sl": sl, "tp": tp, "mb": mb})
    if not best: print("  No profitable configurations found."); return None
    best.sort(key=lambda x: x["ev"], reverse=True)
    b = best[0]
    print(f"  Best: Session({b['sess']}) lookback={b['lb']}h "
          f"SL{b['sl']*100:.2f}%/TP{b['tp']*100:.2f}% max{b['mb']}h")
    print(f"        N={b['n']}  WR={b['wr']:.1f}%  BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")
    rows, passes, mean_ev = walk_forward(
        candles, lambda c, b=b: session_breakout_signals(c, b["lb"], b["sh"]),
        b["sl"], b["tp"], b["mb"])
    print_wf(rows, passes, mean_ev)
    return verdict(passes, mean_ev), passes, mean_ev, b


# ── Strategy 5: MACD Momentum ─────────────────────────────────────────────────
def macd_signals(candles, fast, slow, sig):
    closes = [c["close"] for c in candles]
    fe = ema_series(closes, fast)
    se = ema_series(closes, slow)
    macd_line = [f-s if f and s else None for f,s in zip(fe, se)]
    sig_line  = ema_series(macd_line, sig)
    hist      = [m-s if m is not None and s is not None else None for m,s in zip(macd_line, sig_line)]
    out = []
    for i in range(1, len(candles)):
        if hist[i] is None or hist[i-1] is None or macd_line[i] is None: continue
        if hist[i] > 0 and hist[i-1] <= 0 and macd_line[i] > 0: out.append((i, +1))
        elif hist[i] < 0 and hist[i-1] >= 0 and macd_line[i] < 0: out.append((i, -1))
    return out


def run_macd(candles):
    print("\n  [MACD Momentum]")
    best = []
    for fast, slow, sig in [(5,13,5),(8,21,5),(12,26,9),(12,26,5)]:
        sigs_all = macd_signals(candles, fast, slow, sig)
        for sl in SL_VALUES:
            for tp in TP_VALUES:
                if tp <= sl: continue
                for mb in MAX_BARS_LIST:
                    r = ev_stats(candles, sigs_all, sl, tp, mb)
                    if r and r["ev"] > 0:
                        best.append({**r, "fast": fast, "slow": slow, "sig": sig,
                                     "sl": sl, "tp": tp, "mb": mb})
    if not best: print("  No profitable configurations found."); return None
    best.sort(key=lambda x: x["ev"], reverse=True)
    b = best[0]
    print(f"  Best: MACD({b['fast']},{b['slow']},{b['sig']}) "
          f"SL{b['sl']*100:.2f}%/TP{b['tp']*100:.2f}% max{b['mb']}h")
    print(f"        N={b['n']}  WR={b['wr']:.1f}%  BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")
    rows, passes, mean_ev = walk_forward(
        candles, lambda c, b=b: macd_signals(c, b["fast"], b["slow"], b["sig"]),
        b["sl"], b["tp"], b["mb"])
    print_wf(rows, passes, mean_ev)
    return verdict(passes, mean_ev), passes, mean_ev, b


# ── Strategy 6: Donchian Breakout ─────────────────────────────────────────────
def donchian_signals(candles, period):
    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]
    out = []
    for i in range(period, len(candles)):
        if candles[i]["high"] > max(highs[i-period:i]): out.append((i, +1))
        elif candles[i]["low"] < min(lows[i-period:i]):  out.append((i, -1))
    return out


def run_donchian(candles):
    print("\n  [Donchian Breakout]")
    best = []
    for period in [10, 20, 30, 50]:
        sigs_all = donchian_signals(candles, period)
        for sl in SL_VALUES:
            for tp in TP_VALUES:
                if tp <= sl: continue
                for mb in MAX_BARS_LIST:
                    r = ev_stats(candles, sigs_all, sl, tp, mb)
                    if r and r["ev"] > 0:
                        best.append({**r, "period": period, "sl": sl, "tp": tp, "mb": mb})
    if not best: print("  No profitable configurations found."); return None
    best.sort(key=lambda x: x["ev"], reverse=True)
    b = best[0]
    print(f"  Best: Don({b['period']}) SL{b['sl']*100:.2f}%/TP{b['tp']*100:.2f}% max{b['mb']}h")
    print(f"        N={b['n']}  WR={b['wr']:.1f}%  BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")
    rows, passes, mean_ev = walk_forward(
        candles, lambda c, b=b: donchian_signals(c, b["period"]), b["sl"], b["tp"], b["mb"])
    print_wf(rows, passes, mean_ev)
    return verdict(passes, mean_ev), passes, mean_ev, b


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print(f"EUR/USD STRATEGY RESEARCH  ({SYMBOL}  1h bars)")
    print("=" * 62)
    candles = load_candles()
    t0 = datetime.fromtimestamp(candles[0]["epoch"],  tz=timezone.utc).strftime("%Y-%m-%d")
    t1 = datetime.fromtimestamp(candles[-1]["epoch"], tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"{len(candles):,} bars  |  {t0} to {t1}\n")

    runners = [
        ("EMA+RSI",          run_ema_rsi),
        ("RSI Reversion",    run_rsi_reversion),
        ("Stochastic",       run_stochastic),
        ("Session Breakout", run_session_breakout),
        ("MACD",             run_macd),
        ("Donchian",         run_donchian),
    ]
    results = {}
    for name, fn in runners:
        r = fn(candles)
        results[name] = r

    print(f"\n\n{'='*62}")
    print("SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Strategy':22} {'WF':>5}  {'MeanEV':>8}  Verdict")
    print(f"  {'-'*52}")
    for name, r in sorted(results.items(),
                          key=lambda x: (x[1][1] if x[1] else -1), reverse=True):
        if r is None:
            print(f"  {name:22} {'n/a':>5}  {'n/a':>8}  FAIL")
        else:
            v, passes, mean_ev, _ = r
            ev_str = f"{mean_ev:+.4f}" if mean_ev else "  n/a  "
            print(f"  {name:22} {passes}/3    {ev_str:>8}  {v}")


if __name__ == "__main__":
    main()
