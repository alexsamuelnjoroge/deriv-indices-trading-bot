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
        for dur, unit in [(1,"t"),(2,"t"),(3,"t"),(5,"t"),(1,"m"),(2,"m"),(5,"m")]:
            try:
                pid, payout = await client.propose_contract(
                    symbol=sym, contract_type="DIGITOVER",
                    duration=dur, duration_unit=unit, stake=1.0, barrier="4"
                )
                print(f"{sym}: dur={dur}{unit}  payout={payout:.2f}  BE={1/(1+payout)*100:.1f}%")
                break  # found a working duration
            except Exception as e:
                code = str(e).split(":")[0]
                if "NotAllowed" not in code and "Invalid" not in code:
                    print(f"{sym} {dur}{unit}: {e}")
        else:
            print(f"{sym}: No supported duration found for DIGITOVER")
    await client.disconnect()

asyncio.run(main())
