"""
Quick check: what contract types does the Deriv API support for Crash/Boom symbols?
Run this to find out if Rise/Fall (CALL/PUT) or only Accumulators are available.

Usage: python check_contracts.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from src.api.client import DerivClient

SYMBOLS = ["CRASH1000", "CRASH500", "BOOM1000", "BOOM500"]


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")

    if not token:
        print("ERROR: set DERIV_TOKEN or DERIV_API_TOKEN in your .env file")
        return

    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    for symbol in SYMBOLS:
        print(f"\n{symbol}")
        print("-" * 40)
        try:
            resp = await client._send({
                "contracts_for": symbol,
            })
            available = resp.get("contracts_for", {}).get("available", [])
            if not available:
                print("  No contracts available (symbol may not be on this account)")
                continue

            seen = set()
            for c in available:
                ct = c.get("contract_type", "?")
                cat = c.get("contract_category_display", c.get("contract_category", "?"))
                min_dur = c.get("min_contract_duration", "?")
                max_dur = c.get("max_contract_duration", "?")
                key = (ct, cat)
                if key not in seen:
                    seen.add(key)
                    print(f"  {ct:<20} ({cat}) | duration: {min_dur} - {max_dur}")

        except Exception as e:
            print(f"  ERROR: {e}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
