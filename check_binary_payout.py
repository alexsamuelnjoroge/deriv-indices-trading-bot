"""Check real binary CALL/PUT payouts for Crash/Boom symbols at various tick durations."""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient

SYMBOLS = ["CRASH1000", "CRASH600", "CRASH900", "BOOM600", "BOOM900"]
DURATIONS = [1, 3, 5, 10]


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    print(f"\n{'Symbol':<12} {'Dur':>4}  {'Payout':>8}  {'Profit%':>8}  {'Status'}")
    print("-" * 50)

    for symbol in SYMBOLS:
        for dur in DURATIONS:
            try:
                prop = await client._send({
                    "proposal":       1,
                    "amount":         1.0,
                    "basis":          "stake",
                    "contract_type":  "CALL",
                    "currency":       "USD",
                    "duration":       dur,
                    "duration_unit":  "t",
                    "symbol":         symbol,
                })
                p      = prop.get("proposal", {})
                err    = prop.get("error", {})
                if err:
                    print(f"{symbol:<12} {dur:>3}t  {'N/A':>8}  {'N/A':>8}  ERROR: {err.get('message', err)}")
                    continue
                payout = float(p.get("payout", 0) or 0)
                stake  = float(p.get("ask_price", 1.0) or 1.0)
                profit_pct = (payout - stake) / stake * 100 if stake > 0 else 0
                print(f"{symbol:<12} {dur:>3}t  {payout:>8.4f}  {profit_pct:>7.1f}%  OK")
            except Exception as e:
                print(f"{symbol:<12} {dur:>3}t  {'N/A':>8}  {'N/A':>8}  EXCEPTION: {e}")

    print()
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
