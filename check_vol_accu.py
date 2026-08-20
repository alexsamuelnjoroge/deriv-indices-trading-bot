"""Check which Volatility indices support ACCU and fetch real barrier values."""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient

GROWTH_RATES = [0.03, 0.04]


def implied_breach_rate(ticks_stayed_in: list, max_ticks: int) -> float:
    """Estimate per-tick breach rate from historical contract survival data.

    Each entry = ticks survived before knockout (or >=max_ticks = survived to end).
    breach_rate = total_knockouts / total_tick_exposure
    """
    knockouts = 0
    total_exposure = 0
    for t in ticks_stayed_in:
        capped = min(t, max_ticks)
        if t < max_ticks:
            knockouts += 1
            total_exposure += max(capped, 1)   # at least 1 tick of exposure
        else:
            total_exposure += max_ticks
    if total_exposure == 0:
        return float("nan")
    return knockouts / total_exposure


def baseline_wr(breach_rate: float, hold_ticks: int) -> float:
    return (1.0 - breach_rate) ** hold_ticks


def accu_payout(growth_rate: float, hold_ticks: int) -> float:
    return (1.0 + growth_rate) ** hold_ticks - 1.0


def be_wr(payout: float) -> float:
    return 1.0 / (1.0 + payout)


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    resp     = await client._send({"active_symbols": "brief"})
    all_syms = resp.get("active_symbols", [])
    first    = all_syms[0] if all_syms else {}
    sym_key  = "symbol" if "symbol" in first else "underlying_symbol"

    vol_syms = sorted(
        [s for s in all_syms if any(
            s[sym_key].upper().startswith(k) for k in ("R_", "1HZ")
        )],
        key=lambda s: s[sym_key],
    )
    print(f"Volatility indices found: {[s[sym_key] for s in vol_syms]}\n")

    SEP  = "=" * 85
    THIN = "-" * 85

    print(SEP)
    print(f"  {'Symbol':<12}  gr    barrier_pct   max_ticks  breach%/tick  WR@8t   BE@8t   Edge?")
    print(SEP)

    results = []
    for s in vol_syms:
        symbol = s[sym_key]
        resp2     = await client._send({"contracts_for": symbol})
        available = resp2.get("contracts_for", {}).get("available", [])
        types     = {c.get("contract_type") for c in available}

        if "ACCU" not in types:
            print(f"  {symbol:<12}  --    NO ACCU")
            continue

        for gr in GROWTH_RATES:
            try:
                prop = await client._send({
                    "proposal":          1,
                    "amount":            1.0,
                    "basis":             "stake",
                    "contract_type":     "ACCU",
                    "currency":          "USD",
                    "growth_rate":       gr,
                    "underlying_symbol": symbol,
                })
                p   = prop.get("proposal", {})
                err = prop.get("error", {})
                if err:
                    print(f"  {symbol:<12}  {gr:.0%}  ERROR: {err.get('message', err)}")
                    continue

                cd         = p.get("contract_details", {})
                barrier    = float(cd.get("tick_size_barrier", 0) or 0)
                max_ticks  = int(cd.get("maximum_ticks", 85) or 85)
                tsi        = cd.get("ticks_stayed_in", [])

                br   = implied_breach_rate(tsi, max_ticks) if tsi else float("nan")
                wr8  = baseline_wr(br, 8) * 100
                pay8 = accu_payout(gr, 8)
                bew8 = be_wr(pay8) * 100
                edge = wr8 - bew8   # positive = above breakeven at baseline

                edge_str = f"+{edge:.1f}%" if edge > 0 else f"{edge:.1f}%"
                flag     = "  << POSITIVE" if edge > 1.0 else ""

                print(
                    f"  {symbol:<12}  {gr:.0%}  {barrier:.3e}      {max_ticks:>3}      "
                    f"{br*100:>5.2f}%      {wr8:>5.1f}%  {bew8:>5.1f}%  {edge_str}{flag}"
                )
                results.append({
                    "symbol": symbol, "gr": gr, "barrier": barrier,
                    "breach_rate": br, "wr8": wr8, "be8": bew8, "edge": edge,
                })
            except Exception as e:
                print(f"  {symbol:<12}  {gr:.0%}  EXCEPTION: {e}")

    print(THIN)
    # Sort by edge descending
    best = sorted(results, key=lambda r: r["edge"], reverse=True)[:5]
    print("\nTop 5 by estimated edge at 8 hold-ticks (baseline, no post-spike filter):")
    for r in best:
        print(f"  {r['symbol']:<12} gr={r['gr']:.0%}  breach={r['breach_rate']*100:.2f}%/t  "
              f"WR={r['wr8']:.1f}%  BE={r['be8']:.1f}%  edge={r['edge']:+.2f}%")

    print("\nNote: positive edge at BASELINE means even random entries are +EV.")
    print("BOOM1000 post-spike edge: breach 1.43% vs 3.73% baseline -> +16% above BE.")
    print("Run diagnose_barrier.py on any symbol with breach_rate similar to BOOM1000 (3-4%).")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
