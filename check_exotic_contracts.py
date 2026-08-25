"""
Check which symbols support Multipliers, Turbos, and Vanillas,
and show key contract specs for each.

Usage: python check_exotic_contracts.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from src.api.client import DerivClient

TARGET = {
    "Multipliers": {"MULTUP", "MULTDOWN"},
    "Turbos":      {"TURBOSLONG", "TURBOSSHORT"},
    "Vanillas":    {"VANILLALONGCALL", "VANILLALONGPUT"},
}
ALL_TARGET = set().union(*TARGET.values())


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")

    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    print("Fetching all active symbols...")
    resp        = await client._send({"active_symbols": "brief"})
    all_symbols = resp.get("active_symbols", [])
    sym_key     = "symbol" if "symbol" in all_symbols[0] else "underlying_symbol"

    print(f"Scanning {len(all_symbols)} symbols...\n")

    # Bucket results by category
    found = {cat: [] for cat in TARGET}

    for s in all_symbols:
        symbol = s[sym_key]
        name   = s.get("display_name", symbol)
        try:
            r         = await client._send({"contracts_for": symbol})
            available = r.get("contracts_for", {}).get("available", [])
            types     = {c.get("contract_type") for c in available}
            for cat, wanted in TARGET.items():
                hit = types & wanted
                if hit:
                    # Collect extra specs from the raw contracts list
                    specs = []
                    for c in available:
                        if c.get("contract_type") in hit:
                            specs.append(c)
                    found[cat].append((symbol, name, hit, specs))
        except Exception:
            pass

    SEP = "=" * 70

    for cat, items in found.items():
        print(SEP)
        print(f"  {cat.upper()}  ({len(items)} symbol(s))")
        print(SEP)
        if not items:
            print("  None found on this account.\n")
            continue

        for symbol, name, types, specs in items:
            print(f"\n  {symbol}  ({name})")
            print(f"  Contract types: {', '.join(sorted(types))}")

            # Show relevant fields from the first contract spec of each type
            shown = set()
            for c in specs:
                ct = c.get("contract_type")
                if ct in shown:
                    continue
                shown.add(ct)

                # Multiplier-specific
                if ct in ("MULTUP", "MULTDOWN"):
                    mults    = c.get("multiplier_range", [])
                    min_dur  = c.get("min_contract_duration", "?")
                    max_dur  = c.get("max_contract_duration", "?")
                    stop_out = c.get("stop_out_level", "?")
                    print(f"    [{ct}] multipliers={mults}  stop_out={stop_out}  dur={min_dur}–{max_dur}")

                # Turbo-specific
                elif ct in ("TURBOSLONG", "TURBOSSHORT"):
                    min_dur = c.get("min_contract_duration", "?")
                    max_dur = c.get("max_contract_duration", "?")
                    barriers = c.get("barrier_choices", [])
                    print(f"    [{ct}] dur={min_dur}–{max_dur}  example_barriers={barriers[:4]}")

                # Vanilla-specific
                elif ct in ("VANILLALONGCALL", "VANILLALONGPUT"):
                    min_dur  = c.get("min_contract_duration", "?")
                    max_dur  = c.get("max_contract_duration", "?")
                    strikes  = c.get("barrier_choices", [])
                    print(f"    [{ct}] dur={min_dur}–{max_dur}  example_strikes={strikes[:4]}")

        print()

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
