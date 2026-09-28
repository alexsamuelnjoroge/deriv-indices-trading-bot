import asyncio, os
from dotenv import load_dotenv
from src.api.client import DerivClient

async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    print(f"Token found: {'yes' if token else 'NO — check .env'}")
    client = DerivClient(token)
    await client.connect()
    for sym in ["CRASH500", "CRASH1000", "BOOM500", "BOOM1000"]:
        try:
            pid, payout = await client.propose_contract(
                symbol=sym, contract_type="DIGITOVER",
                duration=1, duration_unit="t", stake=1.0, barrier="4"
            )
            print(f"{sym}: payout={payout:.2f}  BE={1/(1+payout)*100:.1f}%")
        except Exception as e:
            print(f"{sym}: ERROR — {e}")
    await client.disconnect()

asyncio.run(main())
