"""Check which Volatility indices support ACCU and fetch real barrier values."""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient


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

    print(f"{'='*65}")
    print("  Symbol       ACCU?   growth=3%              growth=4%")
    print(f"{'='*65}")

    for s in vol_syms:
        symbol = s[sym_key]
        resp2     = await client._send({"contracts_for": symbol})
        available = resp2.get("contracts_for", {}).get("available", [])
        types     = {c.get("contract_type") for c in available}

        if "ACCU" not in types:
            print(f"  {symbol:<12}  NO")
            continue

        barriers = {}
        _debug_done = False
        for gr in [0.03, 0.04]:
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
                p = prop.get("proposal", {})
                # Debug: dump all keys for first symbol/growth to find barrier fields
                if not _debug_done:
                    _debug_done = True
                    print(f"\n  [DEBUG {symbol} gr={gr}] proposal keys: {list(p.keys())}")
                    # Print any key that looks barrier-related
                    for k, v in p.items():
                        if any(x in k.lower() for x in ("barrier", "spot", "limit", "tick", "high", "low")):
                            print(f"    {k} = {v}")
                    # Also check limit_order or contract_details sub-dicts
                    for k, v in p.items():
                        if isinstance(v, dict):
                            print(f"    {k} (dict) = {v}")
                spot = float(p.get("spot", 0) or 0)
                hb   = float(p.get("high_barrier", 0) or 0)
                # Also try alternative field names
                if hb == 0:
                    hb = float(p.get("barrier",      0) or 0)
                if hb == 0:
                    hb = float(p.get("upper_barrier", 0) or 0)
                if spot > 0 and hb > 0:
                    bp = (hb - spot) / spot
                    barriers[gr] = bp
            except Exception as e:
                barriers[gr] = None
                if not _debug_done:
                    print(f"\n  [ERROR {symbol} gr={gr}]: {e}")

        b3 = f"{barriers.get(0.03):.2e}" if barriers.get(0.03) else "N/A"
        b4 = f"{barriers.get(0.04):.2e}" if barriers.get(0.04) else "N/A"
        print(f"  {symbol:<12}  YES     barrier_pct={b3}     barrier_pct={b4}")

    print()
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
