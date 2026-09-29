import asyncio, os
from dotenv import load_dotenv
from src.api.client import DerivClient

async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    client = DerivClient(token)
    await client.connect()
    for sym in ["R_100", "R_75", "R_50", "R_25", "R_10"]:
        for barrier in ["4", "6", "7"]:
            try:
                pid, payout = await client.propose_contract(
                    symbol=sym, contract_type="DIGITOVER",
                    duration=1, duration_unit="t", stake=1.0, barrier=barrier
                )
                be = 1 / (1 + payout) * 100
                print(f"{sym} DIGITOVER({barrier}): payout={payout*100:.0f}%  BE={be:.1f}%")
            except Exception as e:
                print(f"{sym} DIGITOVER({barrier}): {str(e)[:60]}")
        for barrier in ["4", "3"]:
            try:
                pid, payout = await client.propose_contract(
                    symbol=sym, contract_type="DIGITUNDER",
                    duration=1, duration_unit="t", stake=1.0, barrier=barrier
                )
                be = 1 / (1 + payout) * 100
                print(f"{sym} DIGITUNDER({barrier}): payout={payout*100:.0f}%  BE={be:.1f}%")
            except Exception as e:
                print(f"{sym} DIGITUNDER({barrier}): {str(e)[:60]}")
        print()
    await client.disconnect()

asyncio.run(main())
