"""
Discover all Crash/Boom symbols available on this Deriv account and show
what contract types each one supports.

Usage: python check_contracts.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from src.api.client import DerivClient


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")

    if not token:
        print("ERROR: set DERIV_TOKEN or DERIV_API_TOKEN in your .env file")
        return

    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    # Step 1: fetch all active symbols and filter for Crash/Boom
    print("Fetching active symbols...")
    resp = await client._send({"active_symbols": "brief"})
    all_symbols = resp.get("active_symbols", [])

    if not all_symbols:
        print("No symbols returned.")
        await client.disconnect()
        return

    # Detect the correct key name from the first element
    first = all_symbols[0]
    sym_key = "symbol" if "symbol" in first else (
              "underlying_symbol" if "underlying_symbol" in first else None)
    if sym_key is None:
        print(f"Unknown symbol key. First element keys: {list(first.keys())}")
        await client.disconnect()
        return

    crash_boom = sorted(
        [s for s in all_symbols if any(
            s[sym_key].upper().startswith(k) for k in ("CRASH", "BOOM")
        )],
        key=lambda s: s[sym_key]
    )

    if not crash_boom:
        print("No Crash/Boom symbols found on this account.")
        print(f"Sample symbol keys: {list(first.keys())}")
        await client.disconnect()
        return

    print(f"\nFound {len(crash_boom)} Crash/Boom symbol(s):")
    for s in crash_boom:
        print(f"  {s[sym_key]:<20} {s.get('display_name', s.get('market_display_name', ''))}")

    # Step 2: check contract types and fetch real ACCU barrier per symbol
    print("\n" + "=" * 60)
    print("Contract types + ACCU barrier per symbol:")
    print("=" * 60)

    for s in crash_boom:
        symbol = s[sym_key]
        print(f"\n{symbol}")
        print("-" * 40)
        try:
            resp = await client._send({"contracts_for": symbol})
            available = resp.get("contracts_for", {}).get("available", [])
            if not available:
                print("  No contracts available")
                continue

            seen = set()
            has_accu = False
            for c in available:
                ct  = c.get("contract_type", "?")
                cat = c.get("contract_category_display", c.get("contract_category", "?"))
                key = (ct, cat)
                if key not in seen:
                    seen.add(key)
                    print(f"  {ct:<20} ({cat})")
                if ct == "ACCU":
                    has_accu = True

        except Exception as e:
            print(f"  ERROR (contracts_for): {e}")
            continue

        # Fetch a real ACCU proposal to get the actual barrier
        if has_accu:
            print()
            for growth_rate in [0.01, 0.02, 0.03, 0.04, 0.05]:
                try:
                    prop = await client._send({
                        "proposal": 1,
                        "amount": 1.0,
                        "basis": "stake",
                        "contract_type": "ACCU",
                        "currency": "USD",
                        "growth_rate": growth_rate,
                        "underlying_symbol": symbol,
                    })
                    p = prop.get("proposal", {})
                    spot        = float(p.get("spot", 0) or p.get("current_spot", 0) or 0)
                    high_b      = float(p.get("high_barrier", 0) or 0)
                    low_b       = float(p.get("low_barrier", 0) or 0)
                    if spot > 0 and high_b > 0:
                        barrier_abs = high_b - spot
                        barrier_pct = barrier_abs / spot
                        print(
                            f"  growth={int(growth_rate*100)}%  spot={spot:.4f}"
                            f"  barrier=+/-{barrier_abs:.5f}"
                            f"  ({barrier_pct*100:.6f}% of price)"
                            f"  barrier_pct={barrier_pct:.2e}"
                        )
                    else:
                        # Dig into nested objects for barrier info
                        cd = p.get("contract_details", {}) or {}
                        vp = p.get("validation_params", {}) or {}
                        spot_val = float(p.get("spot", 0) or 0)
                        # Try common barrier key names in nested dicts
                        high = (cd.get("high_barrier") or cd.get("barrier") or
                                vp.get("high_barrier") or vp.get("barrier"))
                        low  = (cd.get("low_barrier") or vp.get("low_barrier"))
                        if high and spot_val > 0:
                            barrier_abs = float(high) - spot_val
                            barrier_pct = abs(barrier_abs) / spot_val
                            print(
                                f"  growth={int(growth_rate*100)}%  spot={spot_val:.4f}"
                                f"  barrier=+/-{abs(barrier_abs):.5f}"
                                f"  ({barrier_pct*100:.6f}% of price)"
                                f"  barrier_pct={barrier_pct:.2e}"
                            )
                        else:
                            # Last resort: dump nested contents (first growth rate only)
                            if growth_rate == 0.04:
                                print(f"  growth=4%  contract_details: {cd}")
                                print(f"  growth=4%  validation_params: {vp}")
                except Exception as e:
                    print(f"  growth={int(growth_rate*100)}%  ERROR: {e}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
