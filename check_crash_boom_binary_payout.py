import asyncio, os
from dotenv import load_dotenv
from src.api.client import DerivClient

async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    client = DerivClient(token)
    await client.connect()
    for sym in ["CRASH1000", "CRASH500", "CRASH300N", "BOOM1000", "BOOM500", "BOOM300N"]:
        for ctype in ["PUT", "CALL"]:
            for dur, unit in [(1,"t"),(2,"t"),(5,"t"),(1,"m")]:
                try:
                    pid, payout = await client.propose_contract(
                        symbol=sym, contract_type=ctype,
                        duration=dur, duration_unit=unit, stake=1.0, barrier=None
                    )
                    be = 1 / (1 + payout) * 100
                    print(f"{sym} {ctype} {dur}{unit}: payout={payout*100:.0f}%  BE={be:.1f}%")
                    break
                except Exception as e:
                    err = str(e)[:40]
                    if "NotAllowed" in err or "Invalid" in err or "must be between" in err:
                        continue
                    print(f"{sym} {ctype} {dur}{unit}: {err}")
                    break
            else:
                print(f"{sym} {ctype}: no valid duration found")
        print()
    await client.disconnect()

asyncio.run(main())
