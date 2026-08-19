"""
Measure real BOOM1000 tick-by-tick moves vs the ACCU barrier.

Answers: why does the backtest show 84% WR but live shows 55%?

Collects N live ticks, computes the per-tick pct_move distribution, and
shows what fraction exceed the barrier — which directly determines the
real knockout probability per tick.

Usage:
  python diagnose_barrier.py                  # 500 ticks, BOOM1000, barrier=2.35e-6
  python diagnose_barrier.py --count 1000
  python diagnose_barrier.py --symbol BOOM600 --barrier 3.95e-6
"""

import argparse
import asyncio
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3"
HOLD_TICKS   = 8
GROWTH_RATE  = 0.04


async def collect_live_ticks(symbol: str, count: int, token: str, app_id: str) -> list[float]:
    import websockets

    url = f"{DERIV_WS_URL}?app_id={app_id}"
    prices: list[float] = []

    print(f"Connecting to Deriv WebSocket ...")
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"authorize": token}))
        auth = json.loads(await ws.recv())
        if "error" in auth:
            print(f"Auth error: {auth['error']}")
            return []

        await ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))
        print(f"Collecting {count} live ticks for {symbol} ...")

        while len(prices) < count:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("msg_type") == "tick":
                p = float(msg["tick"].get("quote", 0))
                if p > 0:
                    prices.append(p)
                    n = len(prices)
                    if n % 50 == 0:
                        print(f"  ... {n}/{count}", end="\r", flush=True)

    print(f"\nCollected {len(prices)} ticks.")
    return prices


def analyse(prices: list[float], barrier: float, hold: int) -> None:
    if len(prices) < 10:
        print("Not enough ticks.")
        return

    moves = []
    for i in range(1, len(prices)):
        prev, curr = prices[i - 1], prices[i]
        if prev > 0:
            moves.append(abs(curr - prev) / prev)

    n = len(moves)
    sorted_moves = sorted(moves)
    median_move  = sorted_moves[n // 2]
    spike_thresh = median_move * 50  # 50x median = spike

    spike_idxs   = [i for i, m in enumerate(moves) if m > spike_thresh]
    normal_moves  = [m for m in moves if m <= spike_thresh]

    avg_price    = sum(prices) / len(prices)
    barrier_abs  = avg_price * barrier

    accu_payout  = (1 + GROWTH_RATE) ** hold - 1
    be_wr        = 100 / (1 + accu_payout)

    SEP = "=" * 60
    print()
    print(SEP)
    print(f"  {prices[0]:.2f} → {prices[-1]:.2f}  |  avg price {avg_price:.4f}")
    print(f"  Barrier : ±{barrier_abs:.6f}  ({barrier*1e6:.2f}e-6, {barrier*100:.6f}% of price)")
    print(f"  Spikes  : {len(spike_idxs)} detected (>{spike_thresh:.2e}, {len(normal_moves)} normal ticks)")
    print(SEP)

    if not normal_moves:
        print("  No normal ticks found.")
        return

    exceed       = [m for m in normal_moves if m > barrier]
    exceed_pct   = len(exceed) / len(normal_moves) * 100
    p_survive_1  = 1 - exceed_pct / 100
    p_survive_n  = p_survive_1 ** hold
    ev           = p_survive_n * accu_payout - (1 - p_survive_n)

    avg_n  = sum(normal_moves) / len(normal_moves)
    med_n  = sorted(normal_moves)[len(normal_moves) // 2]

    print(f"\n  NORMAL tick moves  ({len(normal_moves)} ticks):")
    print(f"    Median  : {med_n:.2e}  ({med_n/barrier:.2f}x barrier)")
    print(f"    Average : {avg_n:.2e}  ({avg_n/barrier:.2f}x barrier)")
    print(f"    > barrier : {len(exceed)}/{len(normal_moves)} = {exceed_pct:.2f}%")

    print(f"\n  ACCU survival simulation  (growth={GROWTH_RATE*100:.0f}%/tick, hold={hold}t):")
    print(f"    P(survive 1 tick)    = {p_survive_1*100:.2f}%")
    print(f"    P(survive {hold} ticks)  = {p_survive_n*100:.1f}%  ← predicted WR")
    print(f"    Breakeven WR         = {be_wr:.1f}%")
    print(f"    EV per stake unit    = {ev:+.4f}  ({'PROFITABLE' if ev > 0 else 'LOSING'})")

    print(f"\n  Tick size distribution (multiples of barrier):")
    for mult in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        t     = barrier * mult
        below = sum(1 for m in normal_moves if m <= t)
        bar   = "#" * (below * 30 // len(normal_moves))
        print(f"    < {mult:5.2f}x ({t:.2e}): {below:5}/{len(normal_moves)} = {below/len(normal_moves)*100:5.1f}%  {bar}")

    # Post-spike analysis
    post_spike: list[float] = []
    for idx in spike_idxs:
        for j in range(1, hold + 3):
            k = idx + j
            if k < len(moves) and moves[k] <= spike_thresh:
                post_spike.append(moves[k])

    print(f"\n  POST-SPIKE ticks (ticks 1-{hold+2} after each spike, {len(post_spike)} samples):")
    if post_spike:
        ps_exceed     = sum(1 for m in post_spike if m > barrier)
        ps_exceed_pct = ps_exceed / len(post_spike) * 100
        avg_ps        = sum(post_spike) / len(post_spike)
        ps_survive_n  = (1 - ps_exceed_pct / 100) ** hold
        ps_ev         = ps_survive_n * accu_payout - (1 - ps_survive_n)
        ratio         = ps_exceed_pct / exceed_pct if exceed_pct > 0 else 0
        print(f"    Avg move        : {avg_ps:.2e}  ({avg_ps/barrier:.2f}x barrier)")
        print(f"    > barrier       : {ps_exceed}/{len(post_spike)} = {ps_exceed_pct:.2f}%"
              f"  ({ratio:.1f}x baseline)")
        print(f"    Predicted WR    : {ps_survive_n*100:.1f}%")
        print(f"    EV per stake    : {ps_ev:+.4f}  ({'PROFITABLE' if ps_ev > 0 else 'LOSING'})")
    else:
        print(f"    No post-spike ticks (no spikes in this sample?)")

    print()
    print(SEP)
    print("  DIAGNOSIS:")
    if exceed_pct < 2:
        verdict = "LOW breach rate — backtest should be accurate. Discrepancy is variance or regime change."
    elif exceed_pct < 6:
        verdict = ("MODERATE breach rate. Historical data may have been sparser (compressed) than live,\n"
                   "  inflating backtest WR. Real edge is smaller than backtested.")
    else:
        verdict = ("HIGH breach rate — barrier is too tight vs real tick sizes.\n"
                   "  ACCU at this barrier has NO edge. Stop trading immediately.")

    print(f"  Baseline breach rate {exceed_pct:.1f}% → {verdict}")

    if post_spike:
        if ps_exceed_pct > exceed_pct * 1.5:
            print(f"\n  POST-SPIKE VOLATILITY ELEVATED ({ps_exceed_pct:.1f}% vs {exceed_pct:.1f}% baseline).")
            print("  Entering right after a spike puts you into a WORSE environment.")
            print("  The recoil hypothesis is wrong — spikes are followed by higher volatility.")
        else:
            print(f"\n  Post-spike volatility normal ({ps_exceed_pct:.1f}% vs {exceed_pct:.1f}% baseline).")
            print("  Spike-entry timing is NOT the cause of losses.")

    print(SEP)
    print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default="BOOM1000")
    parser.add_argument("--count",   type=int, default=500,
                        help="Live ticks to collect (default 500, ~5-10 min)")
    parser.add_argument("--barrier", type=float, default=2.35e-6,
                        help="barrier_pct from check_contracts.py (default 2.35e-6 for BOOM1000 4%%)")
    parser.add_argument("--hold",    type=int, default=HOLD_TICKS,
                        help=f"Hold ticks to simulate (default {HOLD_TICKS})")
    args = parser.parse_args()

    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")

    if not token:
        print("ERROR: set DERIV_TOKEN in .env")
        return

    prices = await collect_live_ticks(args.symbol, args.count, token, app_id)
    analyse(prices, args.barrier, args.hold)


if __name__ == "__main__":
    asyncio.run(main())
