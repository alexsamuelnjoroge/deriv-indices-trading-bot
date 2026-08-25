"""
Check which symbols support Matches/Differs (DIGITMATCH/DIGITDIFF)
and show payout details for each digit (0-9).

Usage: python check_digit_contracts.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from src.api.client import DerivClient

DIGIT_TYPES = {"DIGITMATCH", "DIGITDIFF", "DIGITOVER", "DIGITUNDER",
               "DIGITEVEN", "DIGITODD"}


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")

    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    print("Fetching all active symbols...")
    resp        = await client._send({"active_symbols": "brief"})
    all_symbols = resp.get("active_symbols", [])

    first   = all_symbols[0]
    sym_key = "symbol" if "symbol" in first else "underlying_symbol"

    # Check every symbol for digit contract support
    digit_symbols = []
    print(f"Scanning {len(all_symbols)} symbols for digit contract support...\n")

    for s in all_symbols:
        symbol = s[sym_key]
        try:
            r         = await client._send({"contracts_for": symbol})
            available = r.get("contracts_for", {}).get("available", [])
            types     = {c.get("contract_type") for c in available}
            supported = types & DIGIT_TYPES
            if supported:
                digit_symbols.append((symbol, s.get("display_name", symbol), supported))
        except Exception:
            pass

    if not digit_symbols:
        print("No symbols support digit contracts on this account.")
        await client.disconnect()
        return

    print(f"Found {len(digit_symbols)} symbol(s) with digit contract support:\n")
    SEP = "=" * 65

    for symbol, display, supported in digit_symbols:
        print(SEP)
        print(f"  {symbol}  ({display})")
        print(f"  Supported types: {', '.join(sorted(supported))}")
        print()

        # Fetch payout for DIGITMATCH and DIGITDIFF across all digits
        for contract_type in ["DIGITMATCH", "DIGITDIFF"]:
            if contract_type not in supported:
                continue
            print(f"  --- {contract_type} payouts (digit 0-9) ---")
            payouts = []
            for digit in range(10):
                try:
                    prop = await client._send({
                        "proposal":          1,
                        "amount":            1.0,
                        "basis":             "stake",
                        "contract_type":     contract_type,
                        "currency":          "USD",
                        "duration":          1,
                        "duration_unit":     "t",
                        "underlying_symbol": symbol,
                        "barrier":           str(digit),
                    })
                    p       = prop.get("proposal", {})
                    payout  = float(p.get("payout", 0) or 0)
                    profit  = payout - 1.0
                    pct     = profit * 100
                    payouts.append((digit, pct))
                except Exception as e:
                    payouts.append((digit, None))

            for digit, pct in payouts:
                if pct is None:
                    print(f"    digit {digit}: ERROR")
                else:
                    be = 100 / (1 + pct / 100) if pct > 0 else 999
                    print(f"    digit {digit}: profit={pct:+.1f}%  BE={be:.1f}%")
            print()

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
