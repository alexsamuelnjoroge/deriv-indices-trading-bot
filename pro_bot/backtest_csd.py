"""
Cross-Symbol Divergence (CSD) -- proprietary strategy.

Core thesis:
  Correlated instruments share a common driver (USD strength, risk sentiment).
  When one instrument moves significantly in a direction while its correlated
  partner barely moves, the partner is 'lagging' and will almost always catch up.

  This is a convergence trade: fade the divergence by taking the lagging
  instrument in the direction of the leading instrument's move.

  Pairs tested:
    EURUSD <-> GBPUSD  (both driven by USD, high positive correlation)
    XAUUSD <-> EURUSD  (both inversely driven by USD strength)

  Mechanics:
    1. Compute N-bar cumulative ATR-normalized return for both symbols:
       ret_lead = (lead_close[t] - lead_close[t-N]) / lead_atr[t]
       ret_lag  = (lag_close[t]  - lag_close[t-N])  / lag_atr[t]

    2. Divergence condition:
       |ret_lead| >= min_lead_atr    (lead has moved significantly)
       |ret_lag|  <= max_lag_atr     (lag has NOT moved much)
       ret_lead and ret_lag have different signs OR lag is near zero

    3. Signal: take lag symbol in the DIRECTION of lead's move
       (convergence = lag catches up to lead)

    4. Filter: only fire during London or NY session
    5. Macro filter on the lag symbol (daily EMA direction)

  This strategy is completely different from all prior approaches:
  - It reads TWO symbols simultaneously
  - The 'reason' for the trade is not price action on the traded symbol
    but price action on a correlated symbol
  - Edge comes from cross-market information, not single-symbol patterns
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_1H, CACHE_1D
from pro_bot.indicators import atr as _atr, ema as _ema
from pro_bot.strategies.base import Signal

DAYS = 730

WINDOWS = [
    (0.00, 0.25, 0.50),
    (0.00, 0.50, 0.75),
    (0.00, 0.75, 1.00),
    (0.00, 0.875, 1.00),
]
WINDOW_LABELS = [
    "Window 1 (train Q1,    test Q2)   ",
    "Window 2 (train H1,    test H2p1) ",
    "Window 3 (train 75%,   test Q4)   <- closest to optimisation",
    "Window 4 (train 87.5%, test 12.5%)",
]


# ── Cross-symbol divergence signal ────────────────────────────────────────────

def _align(b1h_a, b1h_b):
    """
    Align two bar series by epoch, returning a list of (epoch, bar_a, bar_b)
    for epochs that appear in BOTH series.
    """
    map_a = {b["epoch"]: b for b in b1h_a}
    map_b = {b["epoch"]: b for b in b1h_b}
    common = sorted(set(map_a) & set(map_b))
    return [(e, map_a[e], map_b[e]) for e in common]


def run_csd(b1h_lead, b1h_lag, b1d_lag, cfg, lag_spread):
    """
    Generate signals on the LAG symbol when the LEAD symbol diverges.
    Signals are returned as (index_into_b1h_lag, Signal) tuples.
    """
    window        = cfg["window_bars"]      # lookback to measure moves
    min_lead_atr  = cfg["min_lead_atr"]     # lead must have moved >= this (ATR units)
    max_lag_atr   = cfg["max_lag_atr"]      # lag must have moved <= this (ATR units)
    cooldown_bars = cfg.get("cooldown_bars", 12)
    session       = cfg.get("session", "both")
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)

    # Pre-compute ATR for both symbols
    atr_lead = _atr(b1h_lead, 14)
    atr_lag  = _atr(b1h_lag,  14)

    # Build epoch-indexed arrays for lead symbol
    lead_close = [b["close"] for b in b1h_lead]
    lead_epoch = [b["epoch"] for b in b1h_lead]
    lead_map   = {e: i for i, e in enumerate(lead_epoch)}

    # Daily macro on lag
    if use_macro and b1d_lag:
        ema_d = _ema([b["close"] for b in b1d_lag], macro_p)
        ep_d  = [b["epoch"] for b in b1d_lag]
    else:
        ema_d = ep_d = None

    # Build a reverse index: lag bar epoch -> lag bar index (for result alignment)
    lag_epoch_to_idx = {b["epoch"]: i for i, b in enumerate(b1h_lag)}

    signals    = []
    last_sig_i = -999

    for j in range(window + 5, len(b1h_lag)):
        lag_bar = b1h_lag[j]
        epoch   = lag_bar["epoch"]
        h       = (epoch % 86400) // 3600

        # Session filter
        in_london = (7 <= h <= 16)
        in_ny     = (13 <= h <= 21)
        if session == "london" and not in_london:
            continue
        if session == "ny" and not in_ny:
            continue
        if session == "both" and not (in_london or in_ny):
            continue

        # Cooldown
        if j - last_sig_i < cooldown_bars:
            continue

        # ATR for lag at this bar
        atr_lag_val = atr_lag[j]
        if atr_lag_val is None or atr_lag_val <= 0:
            continue

        # Find corresponding lead bar index
        if epoch not in lead_map:
            continue
        li = lead_map[epoch]
        if li < window:
            continue

        # Lead ATR
        atr_lead_val = atr_lead[li]
        if atr_lead_val is None or atr_lead_val <= 0:
            continue

        # Compute window-bar moves (ATR-normalized)
        # Lead: how far has the lead moved in last `window` bars?
        lead_ret = (b1h_lead[li]["close"] - b1h_lead[li - window]["close"]) / atr_lead_val
        # Lag: how far has the lag moved in last `window` bars?
        lag_ret  = (lag_bar["close"]       - b1h_lag[j - window]["close"])  / atr_lag_val

        # Divergence check
        # Lead moved significantly, lag has NOT moved much in the same direction
        if lead_ret >= min_lead_atr and abs(lag_ret) <= max_lag_atr:
            # Lead went UP, lag is flat/didn't move up enough → BUY lag (catch-up)
            direction = "BUY"
        elif lead_ret <= -min_lead_atr and abs(lag_ret) <= max_lag_atr:
            # Lead went DOWN, lag is flat → SELL lag (catch-up)
            direction = "SELL"
        else:
            continue

        # Macro filter on the lag symbol
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_lag_val * atr_mult
        tp = sl * tp_rr

        if direction == "BUY" and allow_long:
            signals.append((j, Signal("BUY",  sl_pips=sl, tp_pips=tp,
                                       reason=f"CSD_buy_lead+{lead_ret:.1f}lag{lag_ret:.1f}")))
            last_sig_i = j
        elif direction == "SELL" and allow_short:
            signals.append((j, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                       reason=f"CSD_sell_lead{lead_ret:.1f}lag{lag_ret:.1f}")))
            last_sig_i = j

    return signals


# ── Simulation / stats ────────────────────────────────────────────────────────

def sim(bars, sigs, spread):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0, max_hold_bars=48)


def stats(trades, min_n=5):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins = sum(1 for t in closed if t.result == "WIN")
    n_wr = sum(1 for t in closed if t.result in ("WIN", "LOSS"))
    ev   = sum(t.r_multiple for t in closed) / len(closed)
    return dict(n=len(closed), wr=wins / n_wr if n_wr else 0, ev=ev,
                net_r=sum(t.r_multiple for t in closed))


# ── Walk-forward ──────────────────────────────────────────────────────────────

def _split_sigs(all_sigs, c1, c2):
    return ([(i, s) for i, s in all_sigs if i < c1],
            [(i - c1, s) for i, s in all_sigs if c1 <= i < c2])


def run_wf(b1h_lead, b1h_lag, b1d_lag, cfg, label, lag_spread, verbose=True):
    all_sigs = run_csd(b1h_lead, b1h_lag, b1d_lag, cfg, lag_spread)
    n        = len(b1h_lag)
    passes   = 0

    if verbose:
        print(f"  Config: {label}  [{len(all_sigs)} total signals]")

    for wi, (_, te_pct, ho_pct) in enumerate(WINDOWS):
        c1 = int(n * te_pct)
        c2 = int(n * ho_pct)

        # Split lead bars proportionally too
        n_lead = len(b1h_lead)
        c1_l   = int(n_lead * te_pct)
        c2_l   = int(n_lead * ho_pct)

        tr_sigs, ho_sigs = _split_sigs(all_sigs, c1, c2)

        tr_s = stats(sim(b1h_lag[:c1],    tr_sigs, lag_spread), min_n=5)
        ho_s = stats(sim(b1h_lag[c1:c2], ho_sigs, lag_spread), min_n=3)

        def fmt(s):
            if s is None:
                return "(too few trades)"
            v = "PASS v" if s["ev"] > 0 else "FAIL x"
            return (f"n={s['n']:>3}  WR {s['wr']*100:>5.1f}%  "
                    f"EV {s['ev']:>+.4f}R  Net {s['net_r']:>+5.1f}R  [{v}]")

        passed = ho_s is not None and ho_s["ev"] > 0
        if passed:
            passes += 1

        if verbose:
            tr_d = (b1h_lag[c1-1]["epoch"] - b1h_lag[0]["epoch"]) // 86400 if c1 > 0 else 0
            ho_d = (b1h_lag[c2-1]["epoch"] - b1h_lag[c1]["epoch"]) // 86400 if c2 > c1 else 0
            mk   = " <-" if wi == 2 else ""
            print(f"    {WINDOW_LABELS[wi]}  [train {tr_d}d / test {ho_d}d]{mk}")
            print(f"      Train  : {fmt(tr_s)}")
            print(f"      Holdout: {fmt(ho_s)}")

    if verbose:
        bar     = "#" * passes + "." * (len(WINDOWS) - passes)
        verdict = ("ROBUST"    if passes == 4 else
                   "MOSTLY OK" if passes >= 3 else
                   "MARGINAL"  if passes >= 2 else
                   "OVERFIT -- DO NOT TRADE")
        print(f"\n    [{bar}] {passes}/{len(WINDOWS)} windows positive  -> {verdict}\n")

    return passes


# ── Parameter sweep ───────────────────────────────────────────────────────────

def sweep(b1h_lead, b1h_lag, b1d_lag, train_end_idx_lag, lag_spread):
    lead_cut = int(len(b1h_lead) * 0.60)
    tr_lead  = b1h_lead[:lead_cut]
    tr_lag   = b1h_lag[:train_end_idx_lag]
    results  = []

    for window in [3, 6, 12]:
        for min_lead in [1.0, 1.5, 2.0, 2.5]:
            for max_lag in [0.3, 0.5, 0.8]:
                for session in ["london", "ny", "both"]:
                    for cooldown in [6, 12]:
                        for tp_rr in [1.5, 2.0, 3.0]:
                            for atr_mult in [1.0, 1.5, 2.0]:
                                for macro in [False, True]:
                                    cfg = dict(
                                        window_bars=window,
                                        min_lead_atr=min_lead,
                                        max_lag_atr=max_lag,
                                        session=session,
                                        cooldown_bars=cooldown,
                                        tp_rr=tp_rr,
                                        atr_mult_sl=atr_mult,
                                        macro_filter=macro,
                                        macro_ema_period=20,
                                    )
                                    label = (
                                        f"CSD win{window} lead>={min_lead}ATR lag<={max_lag}ATR "
                                        f"sess={session} cool{cooldown} "
                                        f"RR{tp_rr} ATRx{atr_mult} "
                                        f"{'MACRO' if macro else 'free'}"
                                    )
                                    sigs = run_csd(tr_lead, tr_lag, b1d_lag, cfg, lag_spread)
                                    if not sigs:
                                        continue
                                    trades = sim(tr_lag, sigs, lag_spread)
                                    s = stats(trades, min_n=8)
                                    if s and s["ev"] > 0:
                                        results.append((s["ev"], s["n"], label, cfg))

    results.sort(key=lambda x: -x[0])
    return results


# ── Pair definitions ──────────────────────────────────────────────────────────

PAIRS = [
    # (lead_sym, lag_sym, description)
    ("frxEURUSD", "frxGBPUSD", "EURUSD->GBPUSD"),
    ("frxGBPUSD", "frxEURUSD", "GBPUSD->EURUSD"),
    ("frxXAUUSD", "frxEURUSD", "XAUUSD->EURUSD"),
    ("frxXAUUSD", "frxGBPUSD", "XAUUSD->GBPUSD"),
]


async def main():
    import time as _t

    print("=" * 78)
    print("Cross-Symbol Divergence (CSD) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: correlated pair diverges = lagging symbol catches up to leading one")
    print("Pairs: EURUSD<->GBPUSD (USD correlation), XAUUSD->FX (risk-off USD inverse)")
    print("=" * 78 + "\n")

    all_robust = []

    # Cache all needed data upfront
    data_cache = {}
    for lead_sym, lag_sym, _ in PAIRS:
        for sym in (lead_sym, lag_sym):
            if sym not in data_cache:
                data_cache[sym] = {
                    "1h": await _fetch(sym, 3600,  DAYS, CACHE_1H),
                    "1d": await _fetch(sym, 86400, DAYS, CACHE_1D),
                }

    for lead_sym, lag_sym, pair_name in PAIRS:
        b1h_lead = data_cache[lead_sym]["1h"]
        b1h_lag  = data_cache[lag_sym]["1h"]
        b1d_lag  = data_cache[lag_sym]["1d"]
        lag_spread = SPREADS[lag_sym]

        print(f"\n{'=' * 78}")
        print(f"  Pair: {pair_name}  (signal trades {lag_sym}, spread={lag_spread})")
        print(f"{'=' * 78}")
        fd = _t.strftime("%Y-%m-%d", _t.gmtime(b1h_lag[0]["epoch"]))
        ld = _t.strftime("%Y-%m-%d", _t.gmtime(b1h_lag[-1]["epoch"]))
        print(f"  Lead: {lead_sym} ({len(b1h_lead)} bars) | Lag: {lag_sym} ({len(b1h_lag)} bars)")
        print(f"  Date range: {fd} -> {ld}")

        train_end = int(len(b1h_lag) * 0.60)
        print(f"\n  Phase 1 -- sweep ({train_end} bars / 60%)...")
        ranked = sweep(b1h_lead, b1h_lag, b1d_lag, train_end, lag_spread)

        print(f"  {len(ranked)} configs with positive EV on training data.")
        if not ranked:
            print(f"  No profitable CSD configs found for {pair_name}.\n")
            continue

        print("  Top 5:")
        for ev, n, label, _ in ranked[:5]:
            print(f"    EV {ev:>+.4f}R  n={n:>3}  {label}")

        wf_results = []
        for ev, n, label, cfg in ranked[:5]:
            print(f"\n  {'-' * 70}")
            passes = run_wf(b1h_lead, b1h_lag, b1d_lag, cfg, label, lag_spread, verbose=True)
            wf_results.append((passes, ev, label, cfg))

        robust = [(p, ev, l, c) for p, ev, l, c in wf_results if p >= 3]
        robust.sort(key=lambda x: (-x[0], -x[1]))

        print(f"\n  {pair_name} SUMMARY:")
        if robust:
            for passes, ev, label, cfg in robust:
                bar     = "#" * passes + "." * (4 - passes)
                verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
                print(f"  [{bar}] {passes}/4  {label}")
                print(f"         -> {verdict}  (train EV {ev:+.4f}R)")
                if passes == 4:
                    print(f"  BEST CONFIG: {cfg}")
                all_robust.append((pair_name, lag_sym, passes, ev, label, cfg))
        else:
            print(f"  No CSD config passed 3+ windows for {pair_name}.")

    print("\n" + "=" * 78)
    print("OVERALL ROBUST / MOSTLY OK:")
    print("=" * 78)
    if all_robust:
        for pair_name, lag_sym, passes, ev, label, cfg in sorted(
                all_robust, key=lambda x: (-x[2], -x[3])):
            bar     = "#" * passes + "." * (4 - passes)
            verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
            print(f"  [{bar}] {passes}/4  {pair_name}  {label}")
            print(f"         -> {verdict}  train EV {ev:+.4f}R")
    else:
        print("  No CSD strategy passed 3+ windows on any pair.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
