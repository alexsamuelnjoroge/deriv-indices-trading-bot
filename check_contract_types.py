"""
Check which contract types are available for Boom/Crash symbols.
Run this from the VPS where DERIV_API_TOKEN is set.

python check_contract_types.py
"""
import asyncio
import json
import os
import sys
from src.api.client import DerivClient
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

SYMBOLS = [
    "BOOM50", "CRASH50",
    "BOOM150N", "CRASH150N",
    "BOOM500", "CRASH500",    # known-working baseline for comparison
    "BOOM1000", "CRASH1000",  # known-working baseline
]

WANT = {"CALL", "PUT", "ACCU", "MULTUP", "MULTDOWN"}


async def main():
    token  = os.getenv("DERIV_API_TOKEN", "")
    app_id = int(os.getenv("DERIV_APP_ID", "1089"))
    if not token:
        print("ERROR: DERIV_API_TOKEN not set")
        return

    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    print()
    print("=" * 70)
    print("  Contract type availability by symbol")
    print("=" * 70)

    for sym in SYMBOLS:
        try:
            resp = await client._send({"contracts_for": sym, "currency": "USD"})
            avail = resp.get("contracts_for", {}).get("available", [])
            types = set(c.get("contract_type") for c in avail)
            found = types & WANT
            has_call_put = "CALL" in types and "PUT" in types
            # Get min duration for CALL
            call_entries = [c for c in avail if c.get("contract_type") == "CALL"]
            min_dur = call_entries[0].get("min_contract_duration", "?") if call_entries else "N/A"
            tag = " ✓ CALL/PUT binary available" if has_call_put else " ✗ NO CALL/PUT"
            print(f"  {sym:<15} {tag}   min_dur={min_dur}   all={sorted(found)}")
        except Exception as e:
            print(f"  {sym:<15}  ERROR: {e}")

    await client.disconnect()
    print()


asyncio.run(main())
