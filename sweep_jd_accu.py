"""
Jump Index post-spike ACCU sweep — JD50, JD75, JD100.

Jump indices have bidirectional spikes (up AND down), unlike CRASH (down only)
or BOOM (up only). The ACCU contract is direction-agnostic, making it ideal.

CrashBoomRecoilStrategy is reused directly — after any large jump, the next
few ticks tend to be calm (same recoil logic applies regardless of direction).

Barriers are fetched live from the Deriv API on first run, then cached.
Tick data reuses the 60k-tick cache from data/<symbol>_60000.json.

Walk-forward: 2x30k windows. Minimum 2/2 to report as validated.

Usage:
  python sweep_jd_accu.py
  python sweep_jd_accu.py --symbol JD100
  python sweep_jd_accu.py --fresh-barriers   # re-fetch live barriers
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.data.history import fetch_ticks
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

load_dotenv()
logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

CACHE_DIR    = Path("data")
BARRIER_CACHE = CACHE_DIR / "jd_barriers.json"
WS_URL       = "wss://ws.derivws.com/websockets/v3?app_id=1089"

TICK_COUNT   = 60_000
WINDOWS      = 2
WINDOW_SIZE  = 30_000

# JD spike frequency is roughly JD<N> = spike every N ticks
# JD10/JD25 excluded: too frequent (same reason BOOM50 failed)
SYMBOLS_CFG = {
    "JD50":  {"nominal_freq": 50,  "spike_mult": 10.0, "atr_period": 30},
    "JD75":  {"nominal_freq": 75,  "spike_mult": 10.0, "atr_period": 30},
    "JD100": {"nominal_freq": 100, "spike_mult": 10.0, "atr_period": 50},
}

GROWTH_RATES    = [0.03, 0.04, 0.05]
SETTLE_TICKS    = [0, 3, 5, 8]
HOLD_TICKS_MAP  = {
    "JD50":  [3, 4, 5, 6],          # short holds — spike every 50t
    "JD75":  [4, 5, 6, 8],          # medium holds
    "JD100": [5, 6, 8, 10],         # longer holds OK — spike every 100t
}

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}
MIN_TRADES = 4
SEP = "=" * 100


# ── Barrier fetch ──────────────────────────────────────────────────────────────

async def fetch_barriers_live(symbols: list[str], growth_rates: list[float]) -> dict:
    """Fetch ACCU barrier_pct for each symbol+growth_rate from Deriv API."""
    import websockets
    barriers = {}

    async with websockets.connect(WS_URL, open_timeout=15) as ws:
        for sym in symbols:
            barriers[sym] = {}
            for gr in growth_rates:
                try:
                    payload = {
                        "proposal":          1,
                        "amount":            1.0,
                        "basis":             "stake",
                        "contract_type":     "ACCU",
                        "currency":          "USD",
                        "growth_rate":       gr,
                        "underlying_symbol": sym,
                    }
                    await ws.send(json.dumps({**payload, "req_id": 1}))
                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    msg = json.loads(raw)
                    if msg.get("error"):
                        print(f"  {sym} gr={gr*100:.0f}%: API error — {msg['error']['message']}")
                        continue
                    p    = msg.get("proposal", {})
                    spot = float(p.get("spot", 0) or 0)
                    high = float(p.get("high_barrier", 0) or 0)
                    if spot > 0 and high > 0:
                        bar_pct = (high - spot) / spot
                        barriers[sym][gr] = bar_pct
                        print(f"  {sym} gr={gr*100:.0f}%  barrier_pct={bar_pct:.2e}  (spot={spot:.2f})")
                    else:
                        print(f"  {sym} gr={gr*100:.0f}%: no barrier in response")
                except Exception as e:
                    print(f"  {sym} gr={gr*100:.0f}%: {e}")
    return barriers


def load_or_fetch_barriers(symbols: list[str], growth_rates: list[float],
                            fresh: bool = False) -> dict:
    if BARRIER_CACHE.exists() and not fresh:
        with open(BARRIER_CACHE) as f:
            cached = json.load(f)
        # Check all needed keys exist
        all_present = all(
            sym in cached and str(gr) in cached[sym]
            for sym in symbols for gr in growth_rates
        )
        if all_present:
            # Convert str keys back to float
            return {sym: {float(k): v for k, v in cached[sym].items()}
                    for sym in symbols if sym in cached}

    print("\nFetching live ACCU barriers from Deriv API...")
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    if not token:
        print("ERROR: set DERIV_TOKEN or DERIV_API_TOKEN in .env to fetch live barriers.")
        print("Alternatively, add barrier_pct values manually to data/jd_barriers.json")
        sys.exit(1)

    barriers = asyncio.run(fetch_barriers_live(symbols, growth_rates))
    # Save with string keys (JSON doesn't support float keys)
    save = {sym: {str(gr): v for gr, v in vals.items()} for sym, vals in barriers.items()}
    with open(BARRIER_CACHE, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  Barriers cached to {BARRIER_CACHE}")
    return barriers


# ── Backtest simulation ────────────────────────────────────────────────────────

def payout(gr: float, ht: int) -> float:
    return (1 + gr) ** ht - 1


def be(gr: float, ht: int) -> float:
    return 1.0 / (1.0 + payout(gr, ht))


def run_combo(ticks, sym_cfg: dict, gr: float, ht: int, st: int, bar: float):
    from src.data.tick_store import TickStore

    pay = payout(gr, ht)
    brk = be(gr, ht)

    strategy_cfg = {
        "symbol_type":        "boom",   # JD jumps both ways — use "boom" as neutral
        "spike_mult":         sym_cfg["spike_mult"],
        "atr_period":         sym_cfg["atr_period"],
        "cooldown_ticks":     max(st, 3),
        "loss_cooldown":      2,
        "barrier_pct":        bar,
        "confirm_threshold":  0.5,
        "settle_ticks":       st,
        "hold_ticks":         ht,
        "growth_rate":        gr,
    }

    wins = losses = passes = 0
    for w in range(WINDOWS):
        seg = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
        strategy = CrashBoomRecoilStrategy(strategy_cfg)
        store    = TickStore(max_ticks=500)
        hold_end = -1
        w_wins = w_trades = 0

        for i, tick in enumerate(seg):
            store.add(tick)
            sig = strategy.evaluate(store)
            if sig.action == "BUY_ACCU" and i > hold_end:
                won = True
                for j in range(i + 1, min(i + 1 + ht, len(seg))):
                    prev = float(seg[j - 1]["quote"])
                    curr = float(seg[j]["quote"])
                    if prev > 0 and abs(curr - prev) / prev > bar:
                        won = False
                        break
                if won:
                    w_wins += 1
                w_trades += 1
                hold_end = i + ht

        if w_trades >= MIN_TRADES and (w_wins / w_trades if w_trades else 0) >= brk:
            passes += 1
        wins   += w_wins
        losses += (w_trades - w_wins)

    trades = wins + losses
    return wins, trades, passes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",         help="Run only this symbol (e.g. JD100)")
    parser.add_argument("--fresh-barriers", action="store_true",
                        help="Re-fetch barriers from live API")
    args = parser.parse_args()

    targets = list(SYMBOLS_CFG.keys())
    if args.symbol:
        targets = [s for s in targets if s == args.symbol]

    barriers = load_or_fetch_barriers(targets, GROWTH_RATES, fresh=args.fresh_barriers)

    for sym in targets:
        sym_cfg = SYMBOLS_CFG[sym]
        print()
        print(SEP)
        print(f"  {sym}  |  Jump ACCU Sweep  |  2x{WINDOW_SIZE:,} ticks")
        print(f"  Nominal spike every ~{sym_cfg['nominal_freq']} ticks  |  "
              f"spike_mult={sym_cfg['spike_mult']}  atr={sym_cfg['atr_period']}")
        print(f"  settle_ticks={SETTLE_TICKS}  hold_ticks={HOLD_TICKS_MAP[sym]}  "
              f"growth_rate={[f'{g*100:.0f}%' for g in GROWTH_RATES]}")
        print(SEP)

        ticks = fetch_ticks(sym, TICK_COUNT)
        print(f"  Using {len(ticks):,} ticks")
        print()

        sym_barriers = barriers.get(sym, {})
        if not sym_barriers:
            print(f"  No barriers available for {sym} — skipping sweep.")
            continue

        results = []
        total = len(SETTLE_TICKS) * len(HOLD_TICKS_MAP[sym]) * len(GROWTH_RATES)
        done  = 0

        for st in SETTLE_TICKS:
            for ht in HOLD_TICKS_MAP[sym]:
                for gr in GROWTH_RATES:
                    bar = sym_barriers.get(gr)
                    if bar is None:
                        done += 1
                        continue
                    wins, trades, passes = run_combo(ticks, sym_cfg, gr, ht, st, bar)
                    done += 1
                    if done % 10 == 0:
                        print(f"  ... {done}/{total}")
                    if trades == 0:
                        continue
                    wr  = wins / trades * 100
                    pay = payout(gr, ht)
                    brk = be(gr, ht) * 100
                    ev  = (wr - brk) / 100 * pay * 100
                    results.append((st, gr, ht, bar, pay * 100, brk, wr, ev, trades, passes))

        results.sort(key=lambda r: (-r[9], -r[7]))

        print(f"  {'st':>3}  {'gr':>4}  {'ht':>3}  {'bar':>10}  "
              f"{'pay%':>6}  {'BE%':>5}  {'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
        print(f"  {'-'*3}  {'-'*4}  {'-'*3}  {'-'*10}  "
              f"{'-'*6}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  ----")

        shown = 0
        for r in results:
            st2, gr2, ht2, bar2, pay2, brk2, wr2, ev2, tr2, ps2 = r
            if ps2 == 0 and ev2 < 0 and shown > 30:
                continue
            marker = "***" if ps2 == 2 and ev2 > 0 else ("*" if ps2 >= 1 and ev2 > 0 else "")
            print(f"  {st2:>3}  {gr2*100:.0f}%  {ht2:>3}  {bar2:.2e}  "
                  f"{pay2:>5.1f}%  {brk2:>5.1f}%  {wr2:>5.1f}%  {ev2:>+8.3f}%  "
                  f"{tr2:>6}  {ps2}/{WINDOWS} {marker}")
            shown += 1

        best = next((r for r in results if r[9] == 2 and r[7] > 0), None)
        if best:
            st2, gr2, ht2, bar2, pay2, brk2, wr2, ev2, tr2, ps2 = best
            print()
            print(f"  Best: st={st2}t  gr={gr2*100:.0f}%  ht={ht2}  "
                  f"WR={wr2:.1f}%  BE={brk2:.1f}%  EV={ev2:+.3f}%  passes={ps2}/{WINDOWS}")
            p_safe = 1 - sym_cfg["nominal_freq"] ** -ht2 if sym_cfg["nominal_freq"] > ht2 else 0
            from math import log
            p_no_spike = (1 - 1 / sym_cfg["nominal_freq"]) ** ht2
            print(f"  P(no spike in {ht2}-tick hold) ~= {p_no_spike*100:.1f}%")
        else:
            print()
            print(f"  No 2/2 positive-EV config found for {sym}.")

    print()
    print(SEP)
    print("  *** = 2/2 passes AND EV > 0   * = 1/2 passes AND EV > 0")
    print("  bar = barrier_pct fetched live from Deriv API")


if __name__ == "__main__":
    main()
