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
        print(f"  {symbol} — contracts_for detail")
        print(f"{'='*60}")
        resp      = await client._send({"contracts_for": symbol})
        available = resp.get("contracts_for", {}).get("available", [])

        for c in available:
            ct = c.get("contract_type", "?")
            if ct not in ("CALL", "CALLE", "PUT", "PUTE"):
                continue
            min_dur  = c.get("min_contract_duration", "?")
            max_dur  = c.get("max_contract_duration", "?")
            exp_type = c.get("expiry_type", "?")
            start_tp = c.get("start_type", "?")
            cat      = c.get("contract_category_display", "?")
            print(f"  {ct:<8} | expiry={exp_type:<10} start={start_tp:<10}"
                  f" | min={min_dur}  max={max_dur}  | {cat}")

    await client.disconnect()

    print()
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
