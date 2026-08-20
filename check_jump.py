"""Quick check: are Jump indices available and do they support ACCU?"""
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

    resp    = await client._send({"active_symbols": "brief"})
    all_syms = resp.get("active_symbols", [])
    first   = all_syms[0] if all_syms else {}
    sym_key = "symbol" if "symbol" in first else "underlying_symbol"

    jump_syms = sorted(
        [s for s in all_syms if s[sym_key].upper().startswith("JD")],
        key=lambda s: s[sym_key],
    )
    print(f"Jump indices on account: {[s[sym_key] for s in jump_syms]}")

    if not jump_syms:
        print("None found — Jump indices not available on this account/region.")
        await client.disconnect()
        return

    print()
    for s in jump_syms:
        symbol = s[sym_key]
        print(f"{symbol}  ({s.get('display_name', '')})")
        resp2     = await client._send({"contracts_for": symbol})
        available = resp2.get("contracts_for", {}).get("available", [])
        types     = sorted({c.get("contract_type") for c in available})
        print(f"  Contract types: {types}")

        if "ACCU" in types:
            for gr in [0.03, 0.04]:
                try:
                    prop = await client._send({
                        "proposal":        1,
                        "amount":          1.0,
                        "basis":           "stake",
                        "contract_type":   "ACCU",
                        "currency":        "USD",
                        "growth_rate":     gr,
                        "underlying_symbol": symbol,
                    })
                    p    = prop.get("proposal", {})
                    spot = float(p.get("spot", 0) or 0)
                    hb   = float(p.get("high_barrier", 0) or 0)
                    if spot > 0 and hb > 0:
                        bp = (hb - spot) / spot
                        print(f"  growth={int(gr*100)}%  barrier_pct={bp:.2e}"
                              f"  ({bp*100:.6f}% of price)")
                    else:
                        print(f"  growth={int(gr*100)}%  barrier data missing: {p}")
                except Exception as e:
                    print(f"  growth={int(gr*100)}%  ERROR: {e}")
        print()

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
