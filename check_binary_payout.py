"""Check real binary CALL/PUT payouts for Crash/Boom symbols at various tick durations."""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient

SYMBOLS = ["CRASH1000", "BOOM600"]  # one crash, one boom — enough to see the pattern


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  {symbol} — ALL available contract types")
        print(f"{'='*60}")
        resp      = await client._send({"contracts_for": symbol})
        available = resp.get("contracts_for", {}).get("available", [])

        seen = {}
        for c in available:
            ct = c.get("contract_type", "?")
            if ct in seen:
                continue
            seen[ct] = True
            min_dur  = c.get("min_contract_duration", "?")
            max_dur  = c.get("max_contract_duration", "?")
            exp_type = c.get("expiry_type", "?")
            cat      = c.get("contract_category_display", "?")
            print(f"  {ct:<16} | expiry={exp_type:<12} | min={str(min_dur):<6}  max={max_dur}  | {cat}")

    await client.disconnect()

    print()
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
