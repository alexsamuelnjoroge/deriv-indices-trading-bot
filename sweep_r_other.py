"""
R_ Volatility Indices — Asian / HighLowTicks / Runs Research

Tests three contract families not yet researched on R_10, R_25, R_75, R_100:

  A  ASIAN (ASIANU / ASIAND)
       Win if the arithmetic mean of the next N ticks is above (U) or below (D)
       the entry spot price.  Two modes tested:
         plain   — always bet UP (to measure raw base rate)
         rsi     — bet UP when RSI > 50, DOWN when RSI < 50

  B  TICKHIGH / TICKLOW
       Predict which of 5 sequential ticks will be the highest (TICKHIGH) or
       lowest (TICKLOW).  Tests all 5 tick positions and an RSI-aligned combo
       (uptrend → TICKHIGH(5), downtrend → TICKLOW(5)).

  C  RUNHIGH / RUNLOW
       Win only if all N ticks go strictly in one direction.
       Tests N = 2, 3, 4 for both pure RUNHIGH and RSI-aligned direction.

Payouts fetched live from Deriv API at startup.
Walk-forward: 3 × 10 k ticks per symbol. *** = all 3 windows pass BE AND EV > 0.

Usage:
  python sweep_r_other.py
  python sweep_r_other.py --symbol R_75
  python sweep_r_other.py --approach asian
  python sweep_r_other.py --approach tickhigh
  python sweep_r_other.py --approach runs
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
sys.path.insert(0, ".")

from src.api.client import DerivClient
from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

SYMBOLS      = ["R_10", "R_25", "R_75", "R_100"]
WINDOWS      = 3
WINDOW_SIZE  = 10_000
TOTAL_TICKS  = WINDOWS * WINDOW_SIZE
MIN_TRADES   = 30
RSI_PERIOD   = 14
SEP = "=" * 110

ASIAN_DURS     = [3, 5, 7, 10]
TICK_POSITIONS = [1, 2, 3, 4, 5]
RUN_NS         = [2, 3, 4]

# Payout tables: filled by fetch_payouts(); keys = (dur_or_pos,)
# Fallback = conservative estimates if API unreachable
_ASIAN_PAY:   dict[int, float] = {3: 0.87, 5: 0.87, 7: 0.87, 10: 0.87}
_TICK_PAY:    dict[int, float] = {1: 3.90, 2: 3.90, 3: 3.90, 4: 3.90, 5: 3.90}
_RUN_PAY:     dict[int, float] = {2: 2.80, 3: 6.00, 4: 13.00}


# ── RSI helper ────────────────────────────────────────────────────────────────

def calc_rsi(prices: list[float], period: int) -> float | None:
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(period):
        d = prices[-(i + 1)] - prices[-(i + 2)]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


# ── Payout fetch ──────────────────────────────────────────────────────────────

async def _fetch_payouts() -> None:
    global _ASIAN_PAY, _TICK_PAY, _RUN_PAY

    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    if not token:
        print("  [WARN] No DERIV_TOKEN — using placeholder payouts")
        return

    client = DerivClient(api_token=token, app_id=app_id)
    try:
        await client.connect()
    except Exception as e:
        print(f"  [WARN] API connect failed: {e} — using placeholder payouts")
        return

    probe = "R_75"  # payouts are the same across R_ symbols

    async def proposal(params: dict) -> float:
        """Return profit_pct = payout - 1 for a $1 stake, or 0 on error."""
        try:
            resp = await client._send({
                "proposal": 1, "amount": 1.0, "basis": "stake",
                "currency": "USD", "underlying_symbol": probe,
                **params,
            })
            p = resp.get("proposal", {})
            # payout = total return on $1 if win; ask_price should be ~1.0
            pay = float(p.get("payout", 0) or 0)
            ask = float(p.get("ask_price", 1) or 1)
            return (pay / ask) - 1.0 if ask > 0 and pay > 0 else 0.0
        except Exception:
            return 0.0

    print(f"  Fetching payouts from Deriv API ({probe})...")

    # Asian
    new_asian = {}
    for dur in ASIAN_DURS:
        ppt = await proposal({"contract_type": "ASIANU", "duration": dur, "duration_unit": "t"})
        if ppt > 0:
            new_asian[dur] = ppt
            print(f"    ASIAN  dur={dur:>2}t  profit={ppt*100:.1f}%  BE={100/(1+ppt):.1f}%")
    if new_asian:
        _ASIAN_PAY = new_asian

    # TickHigh (all 5 positions)
    new_tick = {}
    for pos in TICK_POSITIONS:
        ppt = await proposal({
            "contract_type": "TICKHIGH", "duration": 5, "duration_unit": "t",
            "selected_tick": pos,
        })
        if ppt > 0:
            new_tick[pos] = ppt
            print(f"    TICKHIGH pos={pos}  profit={ppt*100:.1f}%  BE={100/(1+ppt):.1f}%")
    if new_tick:
        _TICK_PAY = new_tick

    # RunHigh
    new_run = {}
    for n in RUN_NS:
        ppt = await proposal({"contract_type": "RUNHIGH", "duration": n, "duration_unit": "t"})
        if ppt > 0:
            new_run[n] = ppt
            print(f"    RUNHIGH  n={n}t  profit={ppt*100:.1f}%  BE={100/(1+ppt):.1f}%")
    if new_run:
        _RUN_PAY = new_run

    await client.disconnect()
    print()


def fetch_payouts() -> None:
    asyncio.run(_fetch_payouts())


# ── Simulation helpers ────────────────────────────────────────────────────────

def _wf_stats(
    window_results: list[tuple[int, int]],   # [(wins, total), ...]
    payout_pct: float,
) -> dict:
    be = 100.0 / (1.0 + payout_pct)
    passes = sum(
        1 for wins, total in window_results
        if total > 0 and wins / total * 100 >= be
    )
    all_wins  = sum(w for w, _ in window_results)
    all_total = sum(t for _, t in window_results)
    wr  = all_wins / all_total * 100 if all_total > 0 else 0.0
    ev  = (wr / 100 - be / 100) * payout_pct * 100
    return {
        "wr": wr, "ev": ev, "be": be,
        "trades": all_total, "passes": passes,
        "n_windows": len(window_results),
    }


def _flag(r: dict) -> str:
    if r["passes"] >= r["n_windows"] and r["ev"] > 0:
        return " ***"
    if r["passes"] >= r["n_windows"] - 1 and r["ev"] > 0:
        return " *"
    return ""


# ── Approach A: ASIAN ─────────────────────────────────────────────────────────

def _asian_window(prices: list[float], dur: int, mode: str) -> tuple[int, int]:
    wins = total = 0
    start = RSI_PERIOD + 1 if mode == "rsi" else 0
    for i in range(start, len(prices) - dur):
        entry  = prices[i]
        avg    = sum(prices[i + 1: i + dur + 1]) / dur
        if mode == "plain":
            won = avg > entry          # always bet UP as baseline
        else:
            rsi = calc_rsi(prices[: i + 1], RSI_PERIOD)
            if rsi is None:
                continue
            won = (avg > entry) if rsi >= 50 else (avg < entry)
        wins  += won
        total += 1
    return wins, total


def sweep_asian(symbol: str, splits: list[list[dict]]) -> None:
    print(f"\n  [ASIAN]")
    print(f"  {'dur':>3}  {'mode':>5}  {'pay%':>5}  {'BE%':>5}  "
          f"{'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
    print(f"  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  ----")

    results = []
    for dur in ASIAN_DURS:
        pay = _ASIAN_PAY.get(dur, 0.87)
        for mode in ("plain", "rsi"):
            window_res = []
            for seg in splits:
                prices = [float(t["quote"]) for t in seg]
                w, t = _asian_window(prices, dur, mode)
                window_res.append((w, t))
            r = _wf_stats(window_res, pay)
            r.update({"dur": dur, "mode": mode, "pay": pay})
            results.append(r)

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        flag = _flag(r)
        print(f"  {r['dur']:>3}  {r['mode']:>5}  {r['pay']*100:>4.0f}%  "
              f"{r['be']:>5.1f}%  {r['wr']:>6.1f}%  {r['ev']:>+8.3f}%  "
              f"{r['trades']:>6}  {r['passes']}/{r['n_windows']}{flag}")


# ── Approach B: TICKHIGH / TICKLOW ────────────────────────────────────────────

def _tick_window(
    prices: list[float], selected: int, contract: str, mode: str
) -> tuple[int, int]:
    wins = total = 0
    start = RSI_PERIOD + 1 if mode == "rsi" else 0
    for i in range(start, len(prices) - 5):
        future = prices[i + 1: i + 6]   # 5 ticks after entry

        if mode == "plain":
            ct = contract
        else:
            rsi = calc_rsi(prices[: i + 1], RSI_PERIOD)
            if rsi is None:
                continue
            # trend-aligned: uptrend → TICKHIGH(5), downtrend → TICKLOW(5)
            ct = "TICKHIGH" if rsi >= 50 else "TICKLOW"

        val = future[selected - 1]
        if ct == "TICKHIGH":
            won = val >= max(future)
        else:
            won = val <= min(future)
        wins  += won
        total += 1
    return wins, total


def sweep_tickhigh(symbol: str, splits: list[list[dict]]) -> None:
    print(f"\n  [TICKHIGH / TICKLOW]  (5-tick window)")
    print(f"  {'ct':>8}  {'pos':>3}  {'mode':>5}  {'pay%':>5}  {'BE%':>5}  "
          f"{'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
    print(f"  {'-'*8}  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  ----")

    results = []
    for contract in ("TICKHIGH", "TICKLOW"):
        for pos in TICK_POSITIONS:
            pay = _TICK_PAY.get(pos, 3.90)
            for mode in ("plain", "rsi"):
                if mode == "rsi" and pos != 5:
                    continue  # RSI-aligned only makes sense for pos=5
                window_res = []
                for seg in splits:
                    prices = [float(t["quote"]) for t in seg]
                    w, t = _tick_window(prices, pos, contract, mode)
                    window_res.append((w, t))
                r = _wf_stats(window_res, pay)
                r.update({"ct": contract, "pos": pos, "mode": mode, "pay": pay})
                results.append(r)

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        flag = _flag(r)
        print(f"  {r['ct']:>8}  {r['pos']:>3}  {r['mode']:>5}  {r['pay']*100:>4.0f}%  "
              f"{r['be']:>5.1f}%  {r['wr']:>6.1f}%  {r['ev']:>+8.3f}%  "
              f"{r['trades']:>6}  {r['passes']}/{r['n_windows']}{flag}")


# ── Approach C: RUNHIGH / RUNLOW ──────────────────────────────────────────────

def _run_window(prices: list[float], n: int, mode: str) -> tuple[int, int]:
    wins = total = 0
    start = RSI_PERIOD + 1 if mode == "rsi" else 0
    for i in range(start, len(prices) - n):
        seg = prices[i: i + n + 1]  # n+1 prices = n consecutive intervals

        if mode == "plain":
            # baseline: always bet RUNHIGH (measure base rate)
            direction = "up"
        else:
            rsi = calc_rsi(prices[: i + 1], RSI_PERIOD)
            if rsi is None:
                continue
            direction = "up" if rsi >= 50 else "down"

        if direction == "up":
            won = all(seg[j + 1] > seg[j] for j in range(n))
        else:
            won = all(seg[j + 1] < seg[j] for j in range(n))
        wins  += won
        total += 1
    return wins, total


def sweep_runs(symbol: str, splits: list[list[dict]]) -> None:
    print(f"\n  [RUNHIGH / RUNLOW]")
    print(f"  {'n':>2}  {'mode':>5}  {'pay%':>6}  {'BE%':>5}  "
          f"{'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass  theory%")
    print(f"  {'-'*2}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  ----  -------")

    results = []
    for n in RUN_NS:
        pay = _RUN_PAY.get(n, 0.0)
        if pay <= 0:
            continue
        theory = 0.5 ** n * 100
        for mode in ("plain", "rsi"):
            window_res = []
            for seg in splits:
                prices = [float(t["quote"]) for t in seg]
                w, t = _run_window(prices, n, mode)
                window_res.append((w, t))
            r = _wf_stats(window_res, pay)
            r.update({"n": n, "mode": mode, "pay": pay, "theory": theory})
            results.append(r)

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        flag = _flag(r)
        print(f"  {r['n']:>2}  {r['mode']:>5}  {r['pay']*100:>5.0f}%  "
              f"{r['be']:>5.1f}%  {r['wr']:>6.1f}%  {r['ev']:>+8.3f}%  "
              f"{r['trades']:>6}  {r['passes']}/{r['n_windows']}{flag}  "
              f"{r['theory']:>5.1f}%")


# ── Symbol driver ─────────────────────────────────────────────────────────────

def sweep_symbol(symbol: str, approach: str) -> None:
    print()
    print(SEP)
    print(f"  {symbol}  |  {WINDOWS}×{WINDOW_SIZE:,} ticks")
    print(SEP)

    ticks = fetch_ticks(symbol, count=TOTAL_TICKS)
    ticks = ticks[-TOTAL_TICKS:]
    if len(ticks) < WINDOW_SIZE:
        print(f"  Not enough ticks ({len(ticks)}), skipping.")
        return
    print(f"  Using {len(ticks):,} ticks")

    splits = [ticks[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE] for i in range(WINDOWS)]

    if approach in ("asian", "all"):
        sweep_asian(symbol, splits)
    if approach in ("tickhigh", "all"):
        sweep_tickhigh(symbol, splits)
    if approach in ("runs", "all"):
        sweep_runs(symbol, splits)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="R_ other contract research: Asian / TickHigh / Runs")
    parser.add_argument("--symbol",   default=None,  help="Single symbol (default: all)")
    parser.add_argument("--approach", default="all",
                        choices=["asian", "tickhigh", "runs", "all"],
                        help="Which approach to test")
    parser.add_argument("--no-api",   action="store_true",
                        help="Skip payout fetch (use built-in fallback values)")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    print()
    print("  R_ VOLATILITY INDICES — ASIAN / TICKHIGH / RUNS")
    print("  A: Asian avg vs entry  |  B: Tick-position extreme  |  C: Consecutive runs")
    print()

    if not args.no_api:
        fetch_payouts()
    else:
        print("  Using fallback payouts (--no-api)\n")

    print(f"  Payouts in use:")
    print(f"    ASIAN:    { {d: f'{p*100:.0f}%' for d, p in _ASIAN_PAY.items()} }")
    print(f"    TICKHIGH: { {p: f'{v*100:.0f}%' for p, v in _TICK_PAY.items()} }")
    print(f"    RUNS:     { {n: f'{p*100:.0f}%' for n, p in _RUN_PAY.items()} }")

    for sym in symbols:
        sweep_symbol(sym, args.approach)

    print()
    print(SEP)
    print("  *** = all windows pass BE AND EV>0   * = all-but-one pass AND EV>0")
    print("  plain = always bet UP (base rate)    rsi = RSI-aligned direction")
    print("  theory% = expected WR on random walk (runs only)")


if __name__ == "__main__":
    main()
