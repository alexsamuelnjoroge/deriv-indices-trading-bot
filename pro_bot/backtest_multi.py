"""
Multi-strategy backtest: find the best strategy per symbol.

Tests 4 strategies on XAUUSD, EURUSD, USDJPY:
  1. MTF RSI-Pullback   (current live strategy — baseline)
  2. MACD Histogram     (momentum shift: works in trending + ranging)
  3. Donchian Breakout  (price-based: works in strong trending markets)
  4. BB + RSI<50        (relaxed pullback: catches shallow pullbacks in trends)

For each symbol:
  - Sweeps parameter variants on 80% training data
  - Validates every config with train EV > 0 on 20% holdout
  - Reports the best config that survives BOTH (train AND holdout positive)

Usage:
  python pro_bot/backtest_multi.py
  python pro_bot/backtest_multi.py --symbol=frxXAUUSD
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import (
    load_data, split_bars, simulate_exits,
    SPREADS,
    _fetch, CACHE_5M, CACHE_1H, CACHE_1D, GRAN_5M, GRAN_1H, GRAN_1D,
)
from pro_bot.indicators import (
    ema as _ema, rsi as _rsi, adx as _adx, atr as _atr,
    bollinger as _bb, macd as _macd, donchian as _donchian,
)
from pro_bot.strategies.base import Signal

FILTER_SYM = next((a.split("=")[1] for a in sys.argv if a.startswith("--symbol=")), None)

SYMBOLS = ["frxXAUUSD", "frxEURUSD", "frxUSDJPY"]
LONG_DAYS = 365


async def load_long(symbol: str):
    b5  = await _fetch(symbol, GRAN_5M, LONG_DAYS, CACHE_5M)
    b1h = await _fetch(symbol, GRAN_1H, LONG_DAYS, CACHE_1H)
    b1d = await _fetch(symbol, GRAN_1D, LONG_DAYS, CACHE_1D)
    return b5, b1h, b1d


# ── Session / macro helpers ───────────────────────────────────────────────────

def _in_session(epoch, mode):
    h = (epoch % 86400) // 3600
    m = (epoch % 3600)  // 60
    if mode == "peak":
        london = (7 <= h < 10) or (h == 10 and m < 30)
        ny     = (13 <= h < 16) or (h == 16 and m < 30)
        return london or ny
    if mode == "full":
        return 7 <= h < 20
    return True


def _macro_dirs(bars_1d, macro_ema_p, epochs_1d):
    """Pre-compute per-day macro direction: True=up, False=down."""
    ema_1d = _ema([b["close"] for b in bars_1d], macro_ema_p)
    return ema_1d, epochs_1d


def _allow(ema_1d, epochs_1d, epoch):
    k = bisect.bisect_right(epochs_1d, epoch) - 1
    if k < 1 or ema_1d[k] is None or ema_1d[k - 1] is None:
        return False, False
    macro_up    = ema_1d[k] > ema_1d[k - 1]
    return macro_up, not macro_up  # allow_long, allow_short


def _sl(bars_5m, i, atr_vals, cfg):
    atr_mult = cfg.get("atr_mult_sl", 1.5)
    if atr_vals is not None and atr_vals[i] is not None:
        sl = atr_vals[i] * atr_mult
    else:
        price = bars_5m[i]["close"]
        sl    = abs(price - bars_5m[i]["low"]) + price * 0.0002
    return max(sl, bars_5m[i]["close"] * 0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 1: MTF RSI-Pullback (with optional adaptive threshold)
# ═══════════════════════════════════════════════════════════════════════════════

def run_rsi(bars_5m, bars_1h, bars_1d=None, cfg=None):
    cfg          = cfg or {}
    ema_p        = cfg.get("ema_period",       100)
    slope_b      = cfg.get("slope_bars",         3)
    rsi_p        = cfg.get("rsi_period",        14)
    rsi_entry    = cfg.get("rsi_entry",        35.0)
    tp_rr        = cfg.get("tp_rr",            1.5)
    adx_min      = cfg.get("adx_min",            0)
    sess         = cfg.get("session",         "off")
    atr_mult     = cfg.get("atr_mult_sl",      1.5)
    macro_ema_p  = cfg.get("macro_ema_period",  20)
    adaptive     = cfg.get("adaptive",        False)
    lookback     = cfg.get("rsi_lookback",      50)

    closes_1h  = [b["close"] for b in bars_1h]
    closes_5m  = [b["close"] for b in bars_5m]
    ema_1h     = _ema(closes_1h, ema_p)
    rsi_5m     = _rsi(closes_5m, rsi_p)
    adx_1h     = _adx(bars_1h, 14) if adx_min > 0 else None
    atr_5m     = _atr(bars_5m, 14)
    epochs_1h  = [b["epoch"] for b in bars_1h]

    ema_1d = epochs_1d = None
    if bars_1d:
        ema_1d    = _ema([b["close"] for b in bars_1d], macro_ema_p)
        epochs_1d = [b["epoch"] for b in bars_1d]

    results = []
    for i in range(rsi_p + 2, len(bars_5m)):
        if not _in_session(bars_5m[i]["epoch"], sess):
            continue
        j = bisect.bisect_right(epochs_1h, bars_5m[i]["epoch"]) - 1
        if j < slope_b or j < 0:
            continue
        e_now, e_prev = ema_1h[j], ema_1h[j - slope_b]
        r_now, r_prev = rsi_5m[i], rsi_5m[i - 1]
        if any(x is None for x in [e_now, e_prev, r_now, r_prev]):
            continue
        if adx_min > 0 and adx_1h is not None:
            av = adx_1h[j]
            if av is None or av < adx_min:
                continue

        al = ash = True
        if ema_1d is not None:
            al, ash = _allow(ema_1d, epochs_1d, bars_5m[i]["epoch"])
            if not al and not ash:
                continue

        if adaptive and i >= lookback:
            recent = sorted(v for v in rsi_5m[i - lookback:i] if v is not None)
            if len(recent) >= max(10, lookback // 2):
                thresh = max(rsi_entry, min(50.0, recent[max(0, int(len(recent) * 0.20) - 1)]))
            else:
                thresh = rsi_entry
        else:
            thresh = rsi_entry
        ob = 100.0 - thresh

        sl = _sl(bars_5m, i, atr_5m, cfg)
        tp = sl * tp_rr
        up, dn = e_now > e_prev, e_now < e_prev

        if up and r_prev >= thresh > r_now and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and r_prev <= ob < r_now and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 2: MACD Histogram Reversal
# ═══════════════════════════════════════════════════════════════════════════════

def run_macd(bars_5m, bars_1h, bars_1d=None, cfg=None):
    """
    BUY:  1h EMA up + macro up + 5m MACD histogram was ≤0, now >0  (zero-cross)
          OR histogram turned upward after a dip (recovery mode)
    SELL: mirror
    """
    cfg         = cfg or {}
    ema_p       = cfg.get("ema_period",       100)
    slope_b     = cfg.get("slope_bars",         3)
    macd_fast   = cfg.get("macd_fast",         12)
    macd_slow   = cfg.get("macd_slow",         26)
    macd_sig    = cfg.get("macd_signal",        9)
    tp_rr       = cfg.get("tp_rr",            2.0)
    sess        = cfg.get("session",         "off")
    macro_ema_p = cfg.get("macro_ema_period",  20)
    cross_only  = cfg.get("zero_cross_only", True)  # True=zero-cross; False=recovery too

    closes_1h  = [b["close"] for b in bars_1h]
    closes_5m  = [b["close"] for b in bars_5m]
    ema_1h     = _ema(closes_1h, ema_p)
    atr_5m     = _atr(bars_5m, 14)
    epochs_1h  = [b["epoch"] for b in bars_1h]

    _, _, hist = _macd(closes_5m, macd_fast, macd_slow, macd_sig)

    ema_1d = epochs_1d = None
    if bars_1d:
        ema_1d    = _ema([b["close"] for b in bars_1d], macro_ema_p)
        epochs_1d = [b["epoch"] for b in bars_1d]

    results = []
    warm = macd_slow + macd_sig + 2
    for i in range(warm, len(bars_5m)):
        if hist[i] is None or hist[i - 1] is None:
            continue
        if not _in_session(bars_5m[i]["epoch"], sess):
            continue

        j = bisect.bisect_right(epochs_1h, bars_5m[i]["epoch"]) - 1
        if j < slope_b or j < 0:
            continue
        e_now, e_prev = ema_1h[j], ema_1h[j - slope_b]
        if e_now is None or e_prev is None:
            continue

        al = ash = True
        if ema_1d is not None:
            al, ash = _allow(ema_1d, epochs_1d, bars_5m[i]["epoch"])
            if not al and not ash:
                continue

        if cross_only:
            buy_sig  = hist[i - 1] <= 0 < hist[i]
            sell_sig = hist[i - 1] >= 0 > hist[i]
        else:
            # Also catch histogram recovering from a dip (hist turned up)
            buy_sig  = (hist[i - 1] <= 0 < hist[i]) or (
                        i >= warm + 1 and hist[i - 2] is not None
                        and hist[i - 1] < hist[i - 2] and hist[i] > hist[i - 1]
                        and hist[i] < 0)
            sell_sig = (hist[i - 1] >= 0 > hist[i]) or (
                        i >= warm + 1 and hist[i - 2] is not None
                        and hist[i - 1] > hist[i - 2] and hist[i] < hist[i - 1]
                        and hist[i] > 0)

        sl = _sl(bars_5m, i, atr_5m, cfg)
        tp = sl * tp_rr
        up, dn = e_now > e_prev, e_now < e_prev

        if up and buy_sig  and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and sell_sig and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 3: Donchian Channel Breakout
# ═══════════════════════════════════════════════════════════════════════════════

def run_donchian(bars_5m, bars_1h, bars_1d=None, cfg=None):
    """
    BUY:  macro up + 5m close > N-bar high (breakout above channel)
    SELL: macro down + 5m close < N-bar low
    One signal per bar; 1h EMA optional filter.
    """
    cfg          = cfg or {}
    dc_period    = cfg.get("dc_period",        20)
    tp_rr        = cfg.get("tp_rr",           2.0)
    sess         = cfg.get("session",        "off")
    macro_ema_p  = cfg.get("macro_ema_period",  20)
    use_ema_filt = cfg.get("ema_filter",      True)
    ema_p        = cfg.get("ema_period",       100)
    slope_b      = cfg.get("slope_bars",         3)
    cooldown_b   = cfg.get("cooldown_bars",      6)   # bars to wait after a signal

    closes_5m  = [b["close"] for b in bars_5m]
    atr_5m     = _atr(bars_5m, 14)
    up_ch, lo_ch = _donchian(bars_5m, dc_period)

    epochs_1h  = [b["epoch"] for b in bars_1h]
    ema_1h     = _ema([b["close"] for b in bars_1h], ema_p) if use_ema_filt else None

    ema_1d = epochs_1d = None
    if bars_1d:
        ema_1d    = _ema([b["close"] for b in bars_1d], macro_ema_p)
        epochs_1d = [b["epoch"] for b in bars_1d]

    results  = []
    last_sig = -cooldown_b - 1

    for i in range(dc_period + 1, len(bars_5m)):
        if up_ch[i - 1] is None or lo_ch[i - 1] is None:
            continue
        if not _in_session(bars_5m[i]["epoch"], sess):
            continue
        if i - last_sig <= cooldown_b:
            continue

        al = ash = True
        if ema_1d is not None:
            al, ash = _allow(ema_1d, epochs_1d, bars_5m[i]["epoch"])
            if not al and not ash:
                continue

        ema_up = ema_dn = True
        if use_ema_filt and ema_1h is not None:
            j = bisect.bisect_right(epochs_1h, bars_5m[i]["epoch"]) - 1
            if j >= slope_b:
                en, ep = ema_1h[j], ema_1h[j - slope_b]
                if en is not None and ep is not None:
                    ema_up = en > ep
                    ema_dn = en < ep

        sl = _sl(bars_5m, i, atr_5m, cfg)
        tp = sl * tp_rr
        price = closes_5m[i]

        # Breakout: close exceeds PREVIOUS bar's channel (avoids look-ahead)
        if price > up_ch[i - 1] and al and ema_up:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
            last_sig = i
        elif price < lo_ch[i - 1] and ash and ema_dn:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
            last_sig = i
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 4: Bollinger Band Touch + RSI < 50 (relaxed pullback)
# ═══════════════════════════════════════════════════════════════════════════════

def run_bb_rsi(bars_5m, bars_1h, bars_1d=None, cfg=None):
    """
    BUY:  1h EMA up + macro up + 5m close touches/breaks lower BB + RSI < 50
    SELL: mirror (upper BB + RSI > 50)
    Catches shallow pullbacks in strong trends where RSI never drops to 35.
    """
    cfg         = cfg or {}
    ema_p       = cfg.get("ema_period",       100)
    slope_b     = cfg.get("slope_bars",         3)
    bb_period   = cfg.get("bb_period",         20)
    bb_std      = cfg.get("bb_std",           2.0)
    rsi_p       = cfg.get("rsi_period",        14)
    rsi_thresh  = cfg.get("rsi_thresh",       50.0)   # RSI < this for BUY
    tp_rr       = cfg.get("tp_rr",            1.5)
    sess        = cfg.get("session",         "off")
    macro_ema_p = cfg.get("macro_ema_period",  20)

    closes_5m  = [b["close"] for b in bars_5m]
    closes_1h  = [b["close"] for b in bars_1h]
    ema_1h     = _ema(closes_1h, ema_p)
    rsi_5m     = _rsi(closes_5m, rsi_p)
    bb_5m      = _bb(closes_5m, bb_period, bb_std)
    atr_5m     = _atr(bars_5m, 14)
    epochs_1h  = [b["epoch"] for b in bars_1h]

    ema_1d = epochs_1d = None
    if bars_1d:
        ema_1d    = _ema([b["close"] for b in bars_1d], macro_ema_p)
        epochs_1d = [b["epoch"] for b in bars_1d]

    results = []
    warm = max(rsi_p + 2, bb_period + 1)
    for i in range(warm, len(bars_5m)):
        if bb_5m[i] is None or rsi_5m[i] is None:
            continue
        if not _in_session(bars_5m[i]["epoch"], sess):
            continue

        j = bisect.bisect_right(epochs_1h, bars_5m[i]["epoch"]) - 1
        if j < slope_b or j < 0:
            continue
        e_now, e_prev = ema_1h[j], ema_1h[j - slope_b]
        if e_now is None or e_prev is None:
            continue

        al = ash = True
        if ema_1d is not None:
            al, ash = _allow(ema_1d, epochs_1d, bars_5m[i]["epoch"])
            if not al and not ash:
                continue

        up_b, mid, lo_b = bb_5m[i]
        r = rsi_5m[i]
        sl = _sl(bars_5m, i, atr_5m, cfg)
        tp = sl * tp_rr
        up, dn = e_now > e_prev, e_now < e_prev
        price = closes_5m[i]

        if up and price <= lo_b and r < rsi_thresh and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and price >= up_b and r > (100 - rsi_thresh) and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep grids
# ═══════════════════════════════════════════════════════════════════════════════

def build_grids():
    grids = []

    # 1. RSI-Pullback (baseline + adaptive variants)
    for ema_p in [100, 200]:
        for rsi_e in [35, 40]:
            for adx_m in [0, 20, 25]:
                for tp in [1.5, 2.0]:
                    for sess in ["off", "peak"]:
                        for adaptive in [False, True]:
                            for macro_p in [20, 50]:
                                grids.append(("RSI", run_rsi, {
                                    "ema_period": ema_p, "rsi_entry": rsi_e,
                                    "adx_min": adx_m, "tp_rr": tp,
                                    "session": sess, "adaptive": adaptive,
                                    "macro_ema_period": macro_p, "atr_mult_sl": 1.5,
                                }))

    # 2. MACD Histogram
    for ema_p in [100, 200]:
        for tp in [1.5, 2.0]:
            for sess in ["off", "peak", "full"]:
                for cross_only in [True, False]:
                    for macro_p in [20, 50]:
                        grids.append(("MACD", run_macd, {
                            "ema_period": ema_p, "tp_rr": tp,
                            "session": sess, "zero_cross_only": cross_only,
                            "macro_ema_period": macro_p,
                            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                        }))
    # Fast MACD variant
    for tp in [1.5, 2.0]:
        for macro_p in [20, 50]:
            grids.append(("MACD-fast", run_macd, {
                "ema_period": 100, "tp_rr": tp, "session": "off",
                "zero_cross_only": True, "macro_ema_period": macro_p,
                "macd_fast": 5, "macd_slow": 13, "macd_signal": 5,
            }))

    # 3. Donchian Breakout
    for dc_p in [10, 20, 30]:
        for tp in [2.0, 3.0]:
            for sess in ["off", "peak", "full"]:
                for ema_filt in [True, False]:
                    for macro_p in [20, 50]:
                        grids.append(("Donchian", run_donchian, {
                            "dc_period": dc_p, "tp_rr": tp,
                            "session": sess, "ema_filter": ema_filt,
                            "macro_ema_period": macro_p, "cooldown_bars": 6,
                        }))

    # 4. BB + RSI<50
    for bb_std in [1.5, 2.0]:
        for rsi_t in [50, 55]:
            for tp in [1.5, 2.0]:
                for sess in ["off", "peak"]:
                    for macro_p in [20, 50]:
                        for ema_p in [100, 200]:
                            grids.append(("BB+RSI", run_bb_rsi, {
                                "ema_period": ema_p, "bb_period": 20,
                                "bb_std": bb_std, "rsi_thresh": rsi_t,
                                "tp_rr": tp, "session": sess,
                                "macro_ema_period": macro_p,
                            }))
    return grids


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics + printing
# ═══════════════════════════════════════════════════════════════════════════════

def stats(trades, min_n=20):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins   = sum(1 for t in closed if t.result == "WIN")
    losses = sum(1 for t in closed if t.result == "LOSS")
    be_c   = sum(1 for t in closed if t.result == "BE")
    n_wr   = wins + losses
    wr     = wins / n_wr if n_wr > 0 else 0.0
    ev     = sum(t.r_multiple for t in closed) / len(closed)
    net_r  = sum(t.r_multiple for t in closed)
    return dict(n=len(closed), wins=wins, losses=losses, be=be_c,
                wr=wr, ev=ev, net_r=net_r)


def fmt(s):
    if s is None:
        return "(too few trades)"
    verdict = "STRONG" if s["ev"] > 0.05 else "OK" if s["ev"] > 0 else "NEG"
    return (f"n={s['n']:>3}  W:{s['wins']:>3} L:{s['losses']:>2} BE:{s['be']:>2}  "
            f"WR:{s['wr']*100:>5.1f}%  EV:{s['ev']:>+.3f}R  Net:{s['net_r']:>+5.1f}R  [{verdict}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 82)
    print("MULTI-STRATEGY BACKTEST — find best strategy per symbol (365 days, holdout 20%)")
    print("Strategies: RSI-Pullback | MACD Histogram | Donchian Breakout | BB+RSI<50")
    print("=" * 82)

    grids = build_grids()
    print(f"  {len(grids)} total config variants to test per symbol\n")

    best_per_sym = {}

    for sym in SYMBOLS:
        if FILTER_SYM and sym != FILTER_SYM:
            continue

        spread = SPREADS.get(sym, 0.0)
        print(f"\n{'─'*82}")
        print(f"  {sym}  |  spread={spread}")
        print(f"{'─'*82}")

        try:
            bars_5m, bars_1h, bars_1d = await load_long(sym)
        except Exception as e:
            print(f"  SKIP — {e}")
            continue

        train, holdout = split_bars(bars_5m, bars_1h, bars_1d)
        print(f"  Train: {len(train['5m'])} bars  |  Holdout: {len(holdout['5m'])} bars\n")

        # ── Sweep on training set ────────────────────────────────────────────
        candidates = []   # (strat_name, cfg, train_stats)
        for strat_name, runner, cfg in grids:
            use_1d = bars_1d   # all strategies use macro filter
            sigs   = runner(train["5m"], train["1h"], train["1d"], cfg)
            if not sigs:
                continue
            trd  = simulate_exits(train["5m"], sigs, spread=spread, be_r=1.0)
            tr   = stats(trd, min_n=30)
            if tr and tr["ev"] > 0.0:
                candidates.append((strat_name, cfg, tr))

        print(f"  Training: {len(candidates)} configs with EV > 0 (of {len(grids)} tested)")

        # ── Validate candidates on holdout ───────────────────────────────────
        winners   = []   # both train + holdout positive
        train_only = []  # train positive, holdout negative

        for strat_name, cfg, tr in sorted(candidates, key=lambda x: -x[2]["ev"]):
            runner = next(r for n, r, c in grids if n == strat_name and c is cfg)
            sigs   = runner(holdout["5m"], holdout["1h"], holdout["1d"], cfg)
            if not sigs:
                continue
            trd  = simulate_exits(holdout["5m"], sigs, spread=spread, be_r=1.0)
            ho   = stats(trd, min_n=15)
            if ho is None:
                continue
            if ho["ev"] > 0.0:
                winners.append((strat_name, cfg, tr, ho))
            else:
                train_only.append((strat_name, cfg, tr, ho))

        print(f"  Holdout:  {len(winners)} configs where BOTH train + holdout are positive\n")

        # ── Report winners grouped by strategy type ──────────────────────────
        if winners:
            by_type = {}
            for strat_name, cfg, tr, ho in winners:
                by_type.setdefault(strat_name, []).append((cfg, tr, ho))

            print("  ── Winning configs (train + holdout both positive) ──\n")
            for stype, entries in sorted(by_type.items()):
                best = max(entries, key=lambda x: x[2]["ev"])
                cfg, tr, ho = best
                print(f"  [{stype}]  best holdout EV: {ho['ev']:+.3f}R")
                print(f"    Config: {cfg}")
                print(f"    Train  : {fmt(tr)}")
                print(f"    Holdout: {fmt(ho)}")
                print()

            # Overall best for this symbol
            sym_best = max(winners, key=lambda x: x[3]["ev"])
            sname, cfg, tr, ho = sym_best
            best_per_sym[sym] = (sname, cfg, tr, ho)
            print(f"  ★ BEST for {sym}: [{sname}]  holdout EV {ho['ev']:+.3f}R  WR {ho['wr']*100:.1f}%")

        else:
            print("  No config survived both training and holdout.")
            # Show best training-only result as a fallback reference
            if train_only:
                sname, cfg, tr, ho = max(train_only, key=lambda x: x[2]["ev"])
                print(f"  (Best training config: [{sname}]  train EV {tr['ev']:+.3f}R, "
                      f"holdout EV {ho['ev']:+.3f}R — does not generalise)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*82}")
    print("FINAL RECOMMENDATION — one strategy per symbol")
    print(f"{'='*82}\n")
    if best_per_sym:
        for sym, (sname, cfg, tr, ho) in best_per_sym.items():
            print(f"  {sym}:  [{sname}]")
            print(f"    Config : {cfg}")
            print(f"    Train  : EV {tr['ev']:+.3f}R  WR {tr['wr']*100:.1f}%  n={tr['n']}")
            print(f"    Holdout: EV {ho['ev']:+.3f}R  WR {ho['wr']*100:.1f}%  n={ho['n']}")
            print()
    else:
        print("  No generalising strategy found for any symbol with these 365 days of data.")


if __name__ == "__main__":
    asyncio.run(main())
