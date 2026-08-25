"""
Phase 3 — Turbo barrier knockout analysis

A Turbo contract is priced at approximately its intrinsic value:
  TURBOSLONG  : cost ≈ spot - barrier  (barrier is BELOW spot)
  TURBOSSHORT : cost ≈ barrier - spot  (barrier is ABOVE spot)

At expiry, if the barrier was never touched:
  TURBOSLONG  payout = spot_expiry - barrier
  TURBOSSHORT payout = barrier - spot_expiry  (or 0 if spot > barrier)

If barrier is touched at any point before expiry → total loss (payout = 0).

The normalized return on a TURBOSLONG:
  NR = (payout - cost) / cost
     = (max(S_expiry - B, 0) - (S0 - B)) / (S0 - B)    [no knockout]
     = -1                                                  [knockout]

If Deriv prices turbos fairly: E[NR] = 0 net of their spread.
If E[NR] > 0 consistently   → underpriced → buy-side edge.
If E[NR] < 0 consistently   → Deriv takes > fair value  → negative EV.

Analysis:
  1. Historical knockout rates and expected NR across barrier% × hold-duration combos
  2. Live proposal prices for comparison (absolute vs theoretical fair value)
  3. Walk-forward consistency check (4 windows)

Symbols: R_10, R_25, R_50, R_75, R_100 (confirmed turbo-eligible)

Usage:
  python sweep_turbo.py                     # full analysis
  python sweep_turbo.py --symbol R_75
  python sweep_turbo.py --hist-only         # skip live proposals
  python sweep_turbo.py --live-only         # only fetch current proposals
  python sweep_turbo.py --fresh             # re-download OHLC cache
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
import websockets
from dotenv import load_dotenv

load_dotenv()

WS_URL    = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR = Path("data/scalp")
GRAN      = 300    # 5-min bars

# Turbo-eligible symbols confirmed from check_exotic_contracts.py
SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

# Barrier distances below/above spot to test
BARRIER_PCTS = [0.003, 0.005, 0.010, 0.020, 0.030, 0.050]

# Hold durations in 5-min bars
HOLD_CONFIGS = {
    "1h":   12,
    "4h":   48,
    "1d":  288,
    "7d": 2016,
}

# For live proposals
PROPOSAL_DURATIONS = [
    (1,  "d"),   # 1 day
    (7,  "d"),   # 1 week
    (30, "d"),   # 1 month
]
PROPOSAL_STAKE = 10.0   # USD per proposal

WINDOWS    = 4
MIN_TRADES = 10


# ── Data fetch ────────────────────────────────────────────────────────────────

async def fetch_ohlc(ws, symbol: str, fresh: bool) -> list[dict]:
    cache = CACHE_DIR / f"{symbol}_{GRAN}_ohlc.json"
    if cache.exists() and not fresh:
        with open(cache) as f:
            return json.load(f)
    await ws.send(json.dumps({
        "ticks_history": symbol,
        "style":         "candles",
        "granularity":   GRAN,
        "count":         5000,
        "end":           "latest",
        "req_id":        1,
    }))
    raw = await asyncio.wait_for(ws.recv(), timeout=30)
    msg = json.loads(raw)
    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])
    candles = [
        {
            "epoch": c.get("open_time", c.get("epoch", 0)),
            "open":  float(c["open"]),
            "high":  float(c["high"]),
            "low":   float(c["low"]),
            "close": float(c["close"]),
        }
        for c in msg["candles"]
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(candles, f)
    return candles


async def get_spot(ws, symbol: str, candles: list[dict]) -> float:
    """Return current spot — use last cached candle close as approximation."""
    if candles:
        return candles[-1]["close"]
    # Fallback: fetch one history bar
    await ws.send(json.dumps({
        "ticks_history": symbol,
        "style":         "candles",
        "granularity":   60,
        "count":         1,
        "end":           "latest",
        "req_id":        98,
    }))
    raw = await asyncio.wait_for(ws.recv(), timeout=15)
    msg = json.loads(raw)
    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])
    return float(msg["candles"][-1]["close"])


async def run_live_proposals_via_client(symbol: str) -> None:
    """
    Fetch live Turbo proposals using DerivClient (PAT auth).
    Gets actual barrier_choices from contracts_for, then prices each one.
    """
    sys.path.insert(0, ".")
    from src.api.client import DerivClient

    token  = os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")

    if not token:
        print("  [live] DERIV_API_TOKEN not set — skipping live proposals")
        return

    client = DerivClient(api_token=token, app_id=app_id)
    try:
        await client.connect()
    except Exception as e:
        print(f"  [live] Auth failed: {e}")
        return

    try:
        # Fetch current spot
        ticks = await client._send({"ticks_history": symbol, "style": "candles",
                                    "granularity": 60, "count": 1, "end": "latest"})
        spot = float(ticks["candles"][-1]["close"])

        # Fetch barrier_choices from contracts_for
        cfor = await client._send({"contracts_for": symbol})
        available = cfor.get("contracts_for", {}).get("available", [])
        turbo_cfgs = {c["contract_type"]: c
                      for c in available if "TURBO" in c.get("contract_type", "")}

        if not turbo_cfgs:
            all_types = [c.get("contract_type","") for c in available]
            print(f"  [live] No Turbo contracts on {symbol} | found: {sorted(set(all_types))[:8]}")
            return

        pass

        print(f"\n  Live proposals  |  {symbol}  spot={spot:.2f}")
        print(f"  (Pricing note: stake = ask_price always. Units = stake / dist.")
        print(f"   Payout at expiry = units x (spot_expiry - barrier_abs) if alive.)")
        print()
        print(f"  {'Type':12} {'Barrier':>7} {'Dist%':>6}  {'Units':>9} {'Dist':>9} {'StakeMin':>9}")
        print(f"  {'-'*12} {'-'*7} {'-'*6}  {'-'*9} {'-'*9} {'-'*9}")

        for ct, cfg in sorted(turbo_cfgs.items()):
            barriers = cfg.get("barrier_choices", [])
            min_dur  = cfg.get("min_contract_duration", "1d")
            dur_val  = int("".join(c for c in min_dur if c.isdigit()) or "1")
            dur_unit = "".join(c for c in min_dur if c.isalpha()) or "d"
            # Pick a few representative barriers: nearest, mid, far OTM
            if not barriers:
                continue

            is_long = ct == "TURBOSLONG"
            # For LONG, barriers are below spot; for SHORT, above spot
            barriers_sorted = sorted(barriers, key=float)
            # Pick the 3 closest to spot (smallest distance) — these match
            # the 0.3%, 0.5%, 1.0% zones that showed 4/4 edge in the sim.
            sample = barriers_sorted[:3]

            _first = True  # print full raw proposal once for debug
            for barrier_choice in sample:
                # barrier_choices are distances from spot (positive number).
                # Deriv proposal uses signed relative barrier: "-dist" for LONG, "+dist" for SHORT
                dist      = float(barrier_choice)
                barrier_pct = dist / spot * 100
                intrinsic   = dist                         # = spot × barrier_pct%
                signed_barrier = f"-{barrier_choice}" if is_long else f"+{barrier_choice}"

                try:
                    resp = await client._send({
                        "proposal":          1,
                        "amount":            PROPOSAL_STAKE,
                        "basis":             "stake",
                        "contract_type":     ct,
                        "currency":          "USD",
                        "duration":          1,
                        "duration_unit":     "d",
                        "underlying_symbol": symbol,
                        "barrier":           signed_barrier,
                    })
                    p           = resp.get("proposal", {})
                    _first      = False
                    units       = float(p.get("display_number_of_contracts", 0))
                    min_stk     = float(p.get("min_stake", 0))
                    details     = p.get("contract_details", {})
                    barrier_abs = float(details.get("barrier", 0))
                    print(
                        f"  {ct:12} {signed_barrier:>7} {barrier_pct:>5.1f}%  "
                        f"{units:>9.4f} {abs(spot - barrier_abs):>9.4f} {min_stk:>9.2f}"
                    )
                except Exception as e:
                    print(f"  {ct:12} {signed_barrier:>7}: ERROR {e}")

    finally:
        await client.disconnect()


# ── Historical simulation ─────────────────────────────────────────────────────

def simulate_turbos(candles: list[dict], barrier_pct: float, hold_bars: int):
    """
    For each entry bar i, simulate both TURBOSLONG and TURBOSSHORT:
      - LONG : barrier = close[i] × (1 - barrier_pct), cost = close[i] - barrier
      - SHORT: barrier = close[i] × (1 + barrier_pct), cost = barrier - close[i]

    During the hold period:
      - Long  knockout: any low[j] <= barrier
      - Short knockout: any high[j] >= barrier

    Returns (long_results, short_results) where each is a list of NR values.
      NR = -1 if knocked out
      NR = (payout - cost) / cost otherwise
    """
    long_nr  = []
    short_nr = []

    for i in range(len(candles) - hold_bars - 1):
        S0 = candles[i]["close"]

        # ── LONG ──────────────────────────────────────────────────────────────
        barrier_L = S0 * (1 - barrier_pct)
        cost_L    = S0 - barrier_L   # = S0 * barrier_pct

        ko_L = False
        for j in range(i + 1, i + hold_bars + 1):
            if candles[j]["low"] <= barrier_L:
                ko_L = True
                break

        if ko_L:
            long_nr.append(-1.0)
        else:
            S_exp   = candles[i + hold_bars]["close"]
            payout  = max(S_exp - barrier_L, 0.0)
            nr      = (payout - cost_L) / cost_L if cost_L > 0 else 0
            long_nr.append(nr)

        # ── SHORT ─────────────────────────────────────────────────────────────
        barrier_S = S0 * (1 + barrier_pct)
        cost_S    = barrier_S - S0   # = S0 * barrier_pct

        ko_S = False
        for j in range(i + 1, i + hold_bars + 1):
            if candles[j]["high"] >= barrier_S:
                ko_S = True
                break

        if ko_S:
            short_nr.append(-1.0)
        else:
            S_exp   = candles[i + hold_bars]["close"]
            payout  = max(barrier_S - S_exp, 0.0)
            nr      = (payout - cost_S) / cost_S if cost_S > 0 else 0
            short_nr.append(nr)

    return long_nr, short_nr


def walk_forward_turbo(candles: list[dict], barrier_pct: float, hold_bars: int):
    """
    4-fold walk-forward on simulate_turbos.
    Returns (mean_long_nr, mean_short_nr, ko_rate_long, ko_rate_short, passes_long, passes_short).
    """
    n  = len(candles)
    ws = n // WINDOWS

    all_long  = []
    all_short = []
    pass_l = pass_s = 0

    for w in range(WINDOWS):
        s   = w * ws
        e   = s + ws if w < WINDOWS - 1 else n
        seg = candles[s:e]

        if len(seg) <= hold_bars + 1:
            continue

        l_nr, s_nr = simulate_turbos(seg, barrier_pct, hold_bars)
        if len(l_nr) >= MIN_TRADES:
            mean_l = sum(l_nr) / len(l_nr)
            if mean_l > 0:
                pass_l += 1
            all_long.extend(l_nr)
        if len(s_nr) >= MIN_TRADES:
            mean_s = sum(s_nr) / len(s_nr)
            if mean_s > 0:
                pass_s += 1
            all_short.extend(s_nr)

    def _stats(nrs):
        if not nrs:
            return 0.0, 0.0, 0
        ko   = sum(1 for r in nrs if r <= -1.0)
        mean = sum(nrs) / len(nrs)
        return mean, ko / len(nrs), len(nrs)

    mean_l, ko_l, n_l = _stats(all_long)
    mean_s, ko_s, n_s = _stats(all_short)
    return mean_l, mean_s, ko_l, ko_s, n_l, n_s, pass_l, pass_s




# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="ALL")
    parser.add_argument("--hist-only", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--fresh",     action="store_true")
    args = parser.parse_args()

    symbols = SYMBOLS if args.symbol.upper() == "ALL" else [args.symbol.upper()]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    SEP  = "=" * 82
    THIN = "-" * 82

    async with websockets.connect(WS_URL, open_timeout=15) as ws:

        # Pre-load all candle data
        candle_data: dict[str, list[dict]] = {}
        for sym in symbols:
            print(f"Loading {sym}...", end=" ", flush=True)
            try:
                c = await fetch_ohlc(ws, sym, args.fresh)
                candle_data[sym] = c
                print(f"{len(c)} candles")
            except Exception as e:
                print(f"FAILED: {e}")
                candle_data[sym] = []

        for sym in symbols:
            candles = candle_data.get(sym, [])
            print()
            print(SEP)
            print(f"  {sym}  |  {len(candles)} candles  |  {WINDOWS}-fold walk-forward")
            print(SEP)

            # ── Historical analysis ───────────────────────────────────────────

            if not args.live_only and len(candles) > 100:
                print()
                print("  HISTORICAL SIMULATION  (NR = normalized return = (payout - cost) / cost)")
                print()
                print(f"  {'Hold':>4}  {'Barr%':>6}  {'N':>5}  "
                      f"{'KO_L%':>6}  {'NR_L':>8}  {'WinL':>6}  "
                      f"{'KO_S%':>6}  {'NR_S':>8}  {'WinS':>6}")
                print(f"  {'-'*4}  {'-'*6}  {'-'*5}  "
                      f"{'-'*6}  {'-'*8}  {'-'*6}  "
                      f"{'-'*6}  {'-'*8}  {'-'*6}")

                best: list[tuple] = []

                for label, hold_bars in HOLD_CONFIGS.items():
                    if hold_bars >= len(candles) // WINDOWS - 2:
                        continue

                    for bp in BARRIER_PCTS:
                        (mean_l, mean_s, ko_l, ko_s,
                         n_l, n_s, pass_l, pass_s) = walk_forward_turbo(candles, bp, hold_bars)

                        if n_l < MIN_TRADES and n_s < MIN_TRADES:
                            continue

                        flag = ""
                        if pass_l == WINDOWS and mean_l > 0:
                            flag += " <LONG4/4>"
                        if pass_s == WINDOWS and mean_s > 0:
                            flag += " <SHORT4/4>"

                        print(
                            f"  {label:>4}  {bp*100:>5.1f}%  {n_l:>5}  "
                            f"{ko_l*100:>6.1f}%  {mean_l:>+8.3f}  {pass_l}/{WINDOWS}  "
                            f"{ko_s*100:>6.1f}%  {mean_s:>+8.3f}  {pass_s}/{WINDOWS}"
                            f"{flag}"
                        )

                        if pass_l == WINDOWS and mean_l > 0:
                            best.append(("LONG",  sym, label, bp, mean_l, ko_l, n_l))
                        if pass_s == WINDOWS and mean_s > 0:
                            best.append(("SHORT", sym, label, bp, mean_s, ko_s, n_s))

                if best:
                    print()
                    print("  *** 4/4 robust combos above ***")
                    for b in best:
                        side, _, label, bp, mean, ko, n = b
                        print(f"  {side:5} | hold={label} | barrier={bp*100:.1f}% | "
                              f"KO={ko*100:.1f}% | E[NR]={mean:+.3f} | N={n}")

            # ── Live proposals ────────────────────────────────────────────────

            if not args.hist_only:
                print()
                try:
                    await run_live_proposals_via_client(sym)
                except Exception as e:
                    print(f"  Live proposals FAILED: {e}")

        print()
        print(SEP)
        print("  Legend:")
        print("  NR     = normalized return = (payout - cost) / cost  (KO => NR=-1)")
        print("  KO%    = knockout rate (price hits barrier before expiry)")
        print("  WinL/S = walk-forward windows with positive mean NR")
        print("  EV%    = (turbo_payout / intrinsic_value - 1) × 100%")
        print("  <4/4>  = profitable in all 4 walk-forward windows")
        print()
        print("  Pricing note: Deriv turbo cost ≈ intrinsic value (spot - barrier).")
        print("  E[NR] > 0 means expected payout > intrinsic cost → buy-side edge.")
        print("  E[NR] < 0 means Deriv extracts value above fair → negative EV.")


if __name__ == "__main__":
    asyncio.run(main())
