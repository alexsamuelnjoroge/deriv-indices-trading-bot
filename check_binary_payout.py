"""Check real binary CALL/PUT payouts for Crash/Boom symbols at various tick durations."""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient

SYMBOLS = ["CRASH1000", "CRASH600", "CRASH900", "BOOM600", "BOOM900"]

# (unit, durations_to_try)
DURATIONS_BY_UNIT = [
    ("t", [1, 5, 10]),
    ("m", [1, 2, 5, 15]),
    ("s", [60, 120]),
]


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    print(f"\n{'Symbol':<12} {'Dur':<6}  {'Payout':>8}  {'Profit%':>8}  Status")
    print("-" * 55)

    for symbol in SYMBOLS:
        for unit, durations in DURATIONS_BY_UNIT:
            for dur in durations:
                label = f"{dur}{unit}"
                try:
                    prop = await client._send({
                        "proposal":          1,
                        "amount":            1.0,
                        "basis":             "stake",
                        "contract_type":     "CALL",
                        "currency":          "USD",
                        "duration":          dur,
                        "duration_unit":     unit,
                        "underlying_symbol": symbol,
                    })
                    p   = prop.get("proposal", {})
                    err = prop.get("error", {})
                    if err:
                        print(f"{symbol:<12} {label:<6}  {'N/A':>8}  {'N/A':>8}  ERROR: {err.get('message', err)}")
                        continue
                    payout     = float(p.get("payout", 0) or 0)
                    stake      = float(p.get("ask_price", 1.0) or 1.0)
                    profit_pct = (payout - stake) / stake * 100 if stake > 0 else 0
                    print(f"{symbol:<12} {label:<6}  {payout:>8.4f}  {profit_pct:>7.1f}%  OK")
                except Exception as e:
                    msg = str(e).split(":")[0]  # short error type only
                    print(f"{symbol:<12} {label:<6}  {'N/A':>8}  {'N/A':>8}  {msg}")

    print()
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
