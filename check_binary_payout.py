"""
Check what MULTUP/MULTDOWN multiplier values are available and what the
commission structure looks like for Crash/Boom symbols.
Also probes CALL/PUT availability on Jump indices.
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient

CRASH_BOOM = ["CRASH1000", "CRASH600", "CRASH900", "BOOM600", "BOOM900"]
JUMP       = ["JD10", "JD25", "JD50", "JD75", "JD100"]
MULT_VALUES = [5, 10, 20, 50, 100, 200, 500]
STAKE = 1.0


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    # ── Multiplier check on CRASH/BOOM ──────────────────────────────────
    print("\n" + "="*65)
    print("  MULTUP availability on CRASH/BOOM  (stake=$1)")
    print("="*65)
    print(f"  {'Symbol':<12} {'Mult':>5}  {'Commission':>12}  {'Ask':>8}  Status")
    print("  " + "-"*55)

    for symbol in CRASH_BOOM[:2]:   # sample CRASH1000 + BOOM600 — pattern holds for all
        for mult in MULT_VALUES:
            try:
                prop = await client._send({
                    "proposal":          1,
                    "amount":            STAKE,
                    "basis":             "stake",
                    "contract_type":     "MULTUP",
                    "currency":          "USD",
                    "multiplier":        mult,
                    "underlying_symbol": symbol,
                })
                p   = prop.get("proposal", {})
                err = prop.get("error", {})
                if err:
                    print(f"  {symbol:<12} {mult:>5}x  {'N/A':>12}  {'N/A':>8}  {err.get('code', err.get('message','?'))}")
                    continue
                ask        = float(p.get("ask_price", STAKE) or STAKE)
                commission = float(p.get("commission", 0) or 0)
                print(f"  {symbol:<12} {mult:>5}x  {commission:>12.6f}  {ask:>8.4f}  OK")
            except Exception as e:
                code = str(e).split(":")[0]
                print(f"  {symbol:<12} {mult:>5}x  {'N/A':>12}  {'N/A':>8}  {code}")

    # ── Binary CALL/PUT on Jump indices ─────────────────────────────────
    print("\n" + "="*65)
    print("  CALL/PUT availability on Jump indices")
    print("="*65)
    print(f"  {'Symbol':<8} {'Dur':<6}  {'Payout':>8}  {'Profit%':>8}  Status")
    print("  " + "-"*50)

    for symbol in JUMP:
        for unit, durations in [("t", [1, 5, 10]), ("m", [1, 5])]:
            for dur in durations:
                label = f"{dur}{unit}"
                try:
                    prop = await client._send({
                        "proposal":          1,
                        "amount":            STAKE,
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
                        print(f"  {symbol:<8} {label:<6}  {'N/A':>8}  {'N/A':>8}  {err.get('code','?')}")
                        continue
                    payout     = float(p.get("payout", 0) or 0)
                    stake_r    = float(p.get("ask_price", STAKE) or STAKE)
                    profit_pct = (payout - stake_r) / stake_r * 100 if stake_r > 0 else 0
                    print(f"  {symbol:<8} {label:<6}  {payout:>8.4f}  {profit_pct:>7.1f}%  OK")
                except Exception as e:
                    code = str(e).split(":")[0]
                    print(f"  {symbol:<8} {label:<6}  {'N/A':>8}  {'N/A':>8}  {code}")

    print()
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
