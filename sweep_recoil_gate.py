"""
Recoil-completion gate sweep for ACCU post-spike entries.

Hypothesis: the large directional recoil tick(s) after a spike are what knock
out ACCUs entered too early. By waiting for the first tick that moves AGAINST
the recoil direction (the "counter-recoil"), we enter AFTER the directional
pressure is spent — when subsequent ticks should be small and calm.

For BOOM spike (price jumped UP):
  Recoil ticks  = price falls (negative moves)
  Counter tick  = price rises (positive move) → enter ACCU here

For CRASH spike (price fell DOWN):
  Recoil ticks  = price rises (positive moves)
  Counter tick  = price falls (negative move) → enter ACCU here

Parameters swept:
  min_recoil  : recoil ticks required before counter qualifies [0, 1, 2, 3]
                0 = enter on any counter-recoil (even the very next tick)
                2 = must see 2 down-ticks first (BOOM) before accepting an up-tick
  max_wait    : skip entry if no counter-recoil within N ticks [5, 10, 20, 30]

Baselines included for comparison: fixed settle=0 and settle=3.

Symbols: all 6 active ACCU symbols. Hold ticks from Phase-3 ROBUST configs.
Walk-forward: 4 × 21,500 ticks (86k total). SPIKE_MULT=12x. MIN_TRADES=10.
"""

import sys
from loguru import logger
from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="ERROR")

# Per-symbol ROBUST params from Phase-3 corrected sweep
SYMBOLS = {
    "BOOM50":   {"type": "boom",  "barrier": 2.15e-6, "hold": 12, "gr": 0.04},
    "CRASH50":  {"type": "crash", "barrier": 2.15e-6, "hold": 12, "gr": 0.04},
    "BOOM500":  {"type": "boom",  "barrier": 4.72e-6, "hold": 15, "gr": 0.05},
    "CRASH500": {"type": "crash", "barrier": 4.72e-6, "hold": 8,  "gr": 0.05},
    "BOOM600":  {"type": "boom",  "barrier": 3.94e-6, "hold": 10, "gr": 0.05},
    "CRASH600": {"type": "crash", "barrier": 3.92e-6, "hold": 10, "gr": 0.05},
}

ATR_PERIOD    = 50
SPIKE_MULT    = 12.0
WINDOWS       = 4
WINDOW_SIZE   = 21_500
TICK_COUNT    = WINDOWS * WINDOW_SIZE
MIN_TRADES    = 10

MIN_RECOILS   = [0, 1, 2, 3]
MAX_WAITS     = [5, 10, 20, 30]
FIXED_SETTLES = [0, 3]


def compute_atr(prices, period):
    atrs  = [0.0] * len(prices)
    moves = [0.0] + [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    for i in range(period, len(prices)):
        atrs[i] = sum(moves[i-period+1:i+1]) / period
    return atrs


def _simulate_accu(seg, entry_i, hold_ticks, barrier_pct):
    """Return (survived, ko_tick)."""
    for t in range(1, hold_ticks + 1):
        idx = entry_i + t
        if idx >= len(seg):
            return False, idx
        prev_p = seg[idx - 1]
        curr_p = seg[idx]
        if prev_p <= 0:
            return False, idx
        if abs(curr_p - prev_p) / prev_p > barrier_pct:
            return False, idx
    return True, -1


def run_window(seg, sym_cfg, mode, settle_ticks=0, min_recoil=0, max_wait=20):
    """
    mode: 'fixed'  — enter at spike + settle_ticks + 1
          'recoil' — enter on counter-recoil tick after min_recoil recoil ticks
    """
    is_boom     = sym_cfg["type"] == "boom"
    barrier_pct = sym_cfg["barrier"]
    hold_ticks  = sym_cfg["hold"]
    gr          = sym_cfg["gr"]

    atrs   = compute_atr(seg, ATR_PERIOD)
    payout = (1 + gr) ** hold_ticks - 1
    be     = 1 / (1 + payout)

    wins = losses = 0
    last_exit = -1
    i = ATR_PERIOD + 1
    guard = hold_ticks + max(max_wait, settle_ticks) + 2

    while i < len(seg) - guard:
        if i <= last_exit:
            i += 1
            continue

        if atrs[i] <= 0:
            i += 1
            continue

        move = seg[i] - seg[i-1]
        spike = (is_boom  and move >=  SPIKE_MULT * atrs[i]) or \
                (not is_boom and move <= -SPIKE_MULT * atrs[i])

        if not spike:
            i += 1
            continue

        # ── Find entry index ─────────────────────────────────────────────
        entry_i = None

        if mode == 'fixed':
            entry_i = i + settle_ticks + 1

        else:  # 'recoil'
            recoil_count = 0
            for t in range(1, max_wait + 1):
                idx = i + t
                if idx >= len(seg):
                    break
                tick_move = seg[idx] - seg[idx - 1]
                if is_boom:
                    if tick_move < 0:
                        recoil_count += 1
                    elif tick_move > 0 and recoil_count >= min_recoil:
                        entry_i = idx
                        break
                else:
                    if tick_move > 0:
                        recoil_count += 1
                    elif tick_move < 0 and recoil_count >= min_recoil:
                        entry_i = idx
                        break

        if entry_i is None or entry_i >= len(seg) - hold_ticks:
            i += 1
            continue

        entry_price = seg[entry_i]
        if entry_price <= 0:
            i += 1
            continue

        survived, ko_tick = _simulate_accu(seg, entry_i, hold_ticks, barrier_pct)

        if survived:
            wins += 1
        else:
            losses += 1

        last_exit = (ko_tick if ko_tick > 0 else entry_i + hold_ticks) + 2
        i += 1

    trades = wins + losses
    wr     = wins / trades * 100 if trades > 0 else 0.0
    return wr, trades, be * 100, payout


def sweep_symbol(symbol, sym_cfg, prices):
    results = []

    # ── Baselines ─────────────────────────────────────────────────────────
    for st in FIXED_SETTLES:
        all_wins = all_trades = passes = 0
        wrs = []
        for w in range(WINDOWS):
            seg = prices[w * WINDOW_SIZE: (w+1) * WINDOW_SIZE]
            if len(seg) < 1000:
                continue
            wr, trades, be, pay = run_window(seg, sym_cfg, 'fixed', settle_ticks=st)
            all_wins   += round(wr / 100 * trades)
            all_trades += trades
            wrs.append((wr, trades, be))
            if trades >= MIN_TRADES and wr >= be:
                passes += 1

        if all_trades == 0:
            continue
        wr_total = all_wins / all_trades * 100
        be_val   = wrs[0][2] if wrs else 0
        ev       = (wr_total / 100 - be_val / 100) * pay * 100
        results.append({
            "mode": "fixed", "settle": st, "min_recoil": "-", "max_wait": "-",
            "wr": wr_total, "be": be_val, "ev": ev,
            "trades": all_trades, "passes": passes, "wrs": wrs,
        })

    # ── Recoil gate ───────────────────────────────────────────────────────
    for min_r in MIN_RECOILS:
        for mw in MAX_WAITS:
            all_wins = all_trades = passes = 0
            wrs = []
            for w in range(WINDOWS):
                seg = prices[w * WINDOW_SIZE: (w+1) * WINDOW_SIZE]
                if len(seg) < 1000:
                    continue
                wr, trades, be, pay = run_window(
                    seg, sym_cfg, 'recoil', min_recoil=min_r, max_wait=mw)
                all_wins   += round(wr / 100 * trades)
                all_trades += trades
                wrs.append((wr, trades, be))
                if trades >= MIN_TRADES and wr >= be:
                    passes += 1

            if all_trades == 0:
                continue
            wr_total = all_wins / all_trades * 100
            be_val   = wrs[0][2] if wrs else 0
            ev       = (wr_total / 100 - be_val / 100) * pay * 100
            results.append({
                "mode": "recoil", "settle": "-", "min_recoil": min_r, "max_wait": mw,
                "wr": wr_total, "be": be_val, "ev": ev,
                "trades": all_trades, "passes": passes, "wrs": wrs,
            })

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
    return results


def main():
    print()
    print("=" * 115)
    print("  RECOIL-COMPLETION GATE SWEEP  |  Directional entry timing for ACCU")
    print(f"  {WINDOWS}×{WINDOW_SIZE:,} tick walk-forward | SPIKE={SPIKE_MULT}x | MIN_TRADES={MIN_TRADES}")
    print("  Compares: fixed settle=0, settle=3 vs recoil gate (min_recoil × max_wait)")
    print("=" * 115)

    summary = []

    for symbol, sym_cfg in SYMBOLS.items():
        print(f"\n  Fetching {symbol}...", flush=True)
        try:
            raw = fetch_ticks(symbol, count=TICK_COUNT)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        prices = [float(t["quote"]) if isinstance(t, dict) else float(t) for t in raw]
        if len(prices) < WINDOW_SIZE * 2:
            print(f"  {symbol}: only {len(prices):,} ticks — skip")
            continue

        results = sweep_symbol(symbol, sym_cfg, prices)

        hold = sym_cfg["hold"]
        gr   = sym_cfg["gr"]
        pay  = (1 + gr) ** hold - 1

        print(f"\n  {'='*105}")
        print(f"  {symbol}  |  {len(prices):,} ticks  |  hold={hold}  gr={gr*100:.0f}%  "
              f"payout={pay*100:.1f}%  BE={100/(1+pay):.1f}%")
        print(f"  {'='*105}")
        print(f"  {'mode':<8}  {'st':>3}  {'minR':>5}  {'maxW':>5}  "
              f"{'WR%':>6}  {'BE%':>5}  {'EV%':>8}  {'trades':>7}  pass  windows")
        print(f"  {'-'*8}  {'-'*3}  {'-'*5}  {'-'*5}  "
              f"{'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  ----  {'-'*35}")

        shown = 0
        best_recoil = None
        for r in results:
            if r["trades"] < MIN_TRADES:
                continue
            flag = ("  ROBUST" if r["passes"] == WINDOWS and r["ev"] > 0 else
                    "  OK"     if r["passes"] == WINDOWS - 1 and r["ev"] > 0 else "")
            wrs_str = "  ".join(
                f"{w:.0f}%({'OK' if w >= b else 'NO'})"
                for w, t, b in r["wrs"] if t >= MIN_TRADES
            )
            st_str = str(r["settle"]) if r["settle"] != "-" else " -"
            mr_str = str(r["min_recoil"]) if r["min_recoil"] != "-" else " -"
            mw_str = str(r["max_wait"])   if r["max_wait"]   != "-" else " -"
            print(f"  {r['mode']:<8}  {st_str:>3}  {mr_str:>5}  {mw_str:>5}  "
                  f"{r['wr']:>5.1f}%  {r['be']:>4.1f}%  "
                  f"{r['ev']:>+7.3f}%  {r['trades']:>7}  "
                  f"{r['passes']}/{WINDOWS}{flag}  [{wrs_str}]")
            shown += 1
            if r["mode"] == "recoil" and best_recoil is None and r["ev"] > 0:
                best_recoil = r
            if shown >= 25:
                remaining = sum(1 for x in results if x["trades"] >= MIN_TRADES) - shown
                if remaining > 0:
                    print(f"  ... ({remaining} more not shown)")
                break

        # Best recoil vs baseline settle=0
        baseline = next((r for r in results
                         if r["mode"] == "fixed" and r["settle"] == 0
                         and r["trades"] >= MIN_TRADES), None)
        if best_recoil and baseline:
            wr_gain = best_recoil["wr"] - baseline["wr"]
            ev_gain = best_recoil["ev"] - baseline["ev"]
            print(f"\n  Best recoil gate: minR={best_recoil['min_recoil']}  "
                  f"maxW={best_recoil['max_wait']}  "
                  f"WR={best_recoil['wr']:.1f}%  EV={best_recoil['ev']:+.3f}%  "
                  f"passes={best_recoil['passes']}/{WINDOWS}")
            print(f"  vs baseline settle=0:  WR={baseline['wr']:.1f}%  "
                  f"EV={baseline['ev']:+.3f}%  "
                  f"  dWR={wr_gain:+.1f}pp  dEV={ev_gain:+.3f}pp")

        robust = [r for r in results
                  if r["passes"] == WINDOWS and r["ev"] > 0
                  and r["trades"] >= MIN_TRADES and r["mode"] == "recoil"]
        if robust:
            summary.append({"symbol": symbol, **robust[0], "sym_cfg": sym_cfg})

    if summary:
        print()
        print("=" * 90)
        print("  SUMMARY — recoil gate 4/4 ROBUST configs")
        print("=" * 90)
        print(f"  {'Symbol':<12}  {'minR':>5}  {'maxW':>5}  {'WR%':>6}  "
              f"{'BE%':>5}  {'EV%':>8}  {'trades':>7}  hold  barrier")
        print(f"  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*6}  "
              f"{'-'*5}  {'-'*8}  {'-'*7}  ----  -------")
        for s in sorted(summary, key=lambda x: x["ev"], reverse=True):
            cfg = s["sym_cfg"]
            print(f"  {s['symbol']:<12}  {s['min_recoil']:>5}  {s['max_wait']:>5}  "
                  f"{s['wr']:>5.1f}%  {s['be']:>4.1f}%  "
                  f"{s['ev']:>+7.3f}%  {s['trades']:>7}  "
                  f"{cfg['hold']:>4}  {cfg['barrier']:.2e}")
    print()


if __name__ == "__main__":
    main()
