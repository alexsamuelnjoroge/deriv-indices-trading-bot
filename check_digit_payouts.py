"""
Probe DIGITDIFF/DIGITMATCH/OVER/UNDER payouts for each digit 0-9.
Correct API parameter: barrier="<digit>"

Key question: does Deriv price each digit differently based on frequency?
If DIGITDIFF pays the same for all digits AND distribution is uniform,
we may have a large edge.

Usage:
  python check_digit_payouts.py 2>/dev/null
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient

SYMBOLS = ["R_10", "R_50", "R_100", "1HZ10V", "1HZ100V", "JD10"]
STAKE   = 1.0
SEP     = "=" * 80
THIN    = "-" * 80


async def proposal(client, symbol, ct, barrier):
    req = {
        "proposal": 1, "amount": STAKE, "basis": "stake", "currency": "USD",
        "contract_type": ct, "underlying_symbol": symbol,
        "duration": 1, "duration_unit": "t",
    }
    if barrier is not None:
        req["barrier"] = str(barrier)
    try:
        resp = await client._send(req)
        p    = resp.get("proposal", {})
        err  = resp.get("error", {})
        if err:
            return None, err.get("code", "?"), err.get("message", "?")[:60]
        payout = float(p.get("payout", 0) or 0)
        ask    = float(p.get("ask_price", STAKE) or STAKE)
        profit = (payout - ask) / ask * 100 if ask > 0 else 0
        be     = 100 / (1 + profit / 100) if profit > 0 else 0
        return {"payout": payout, "ask": ask, "profit": profit, "be": be}, None, None
    except Exception as e:
        return None, "EXCEPTION", str(e)[:60]


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    for symbol in SYMBOLS:
        print()
        print(SEP)
        print(f"  {symbol}")
        print(THIN)

        # DIGITDIFF and DIGITMATCH for each digit 0-9
        for ct in ["DIGITDIFF", "DIGITMATCH"]:
            print(f"\n  {ct}:")
            print(f"  {'Digit':>5}  {'Payout':>8}  {'Profit%':>8}  {'BE%':>6}  Note")
            total_payout = 0
            uniform_wr = 0
            rows = []
            for d in range(10):
                r, code, msg = await proposal(client, symbol, ct, d)
                if r:
                    rows.append((d, r["payout"], r["profit"], r["be"]))
                    total_payout += r["profit"]
                else:
                    rows.append((d, None, None, None))
                    print(f"  {d:>5}  ERROR [{code}]: {msg}")

            for d, payout, profit, be in rows:
                if payout is None:
                    continue
                print(f"  {d:>5}  {payout:>8.4f}  {profit:>7.1f}%  {be:>5.1f}%")

            valid = [(d, p, pr, be) for d, p, pr, be in rows if p is not None]
            if valid:
                # For uniform distribution: actual WR
                if ct == "DIGITDIFF":
                    actual_wr = 90.0   # P(last digit ≠ chosen) for uniform
                else:
                    actual_wr = 10.0   # P(last digit = chosen) for uniform
                avg_profit = sum(pr for _, _, pr, _ in valid) / len(valid)
                avg_be     = sum(be for _, _, _, be in valid) / len(valid)
                print(THIN)
                print(f"  Average payout:  profit={avg_profit:.1f}%  BE={avg_be:.1f}%")
                print(f"  If uniform dist: WR={actual_wr:.0f}%  edge={actual_wr - avg_be:+.1f}%")

        # DIGITOVER and DIGITUNDER (digit means threshold)
        for ct, digits in [("DIGITOVER", [4, 5, 6, 7, 8]),
                            ("DIGITUNDER", [1, 2, 3, 4, 5])]:
            print(f"\n  {ct}:")
            print(f"  {'Digit':>5}  {'Payout':>8}  {'Profit%':>8}  {'BE%':>6}  {'Uniform WR':>10}  {'Edge':>8}")
            for d in digits:
                r, code, msg = await proposal(client, symbol, ct, d)
                if r:
                    # Uniform WR: P(last digit > d) for OVER, P(last digit < d) for UNDER
                    if ct == "DIGITOVER":
                        uniform_wr = (9 - d) * 10.0   # digits d+1..9, 10% each
                    else:
                        uniform_wr = d * 10.0           # digits 0..d-1, 10% each
                    edge = uniform_wr - r["be"]
                    print(f"  {d:>5}  {r['payout']:>8.4f}  {r['profit']:>7.1f}%  "
                          f"{r['be']:>5.1f}%  {uniform_wr:>9.0f}%  {edge:>+7.1f}%")
                else:
                    print(f"  {d:>5}  ERROR [{code}]: {msg}")

        # DIGITEVEN / DIGITODD
        print(f"\n  DIGITEVEN / DIGITODD:")
        for ct in ["DIGITEVEN", "DIGITODD"]:
            r, code, msg = await proposal(client, symbol, ct, None)
            if r:
                # Uniform: 5 even digits (0,2,4,6,8), 5 odd (1,3,5,7,9)
                uniform_wr = 50.0
                edge = uniform_wr - r["be"]
                print(f"  {ct:<12}  payout={r['payout']:.4f}  profit={r['profit']:.1f}%  "
                      f"BE={r['be']:.1f}%  uniform_WR=50%  edge={edge:+.1f}%")
            else:
                print(f"  {ct:<12}  ERROR [{code}]: {msg}")

    print()
    await client.disconnect()
    print("Done.")
    print()
    print("INTERPRETATION:")
    print("  DIGITDIFF BE ≈ 51.3%: If digits are uniform (10% each),")
    print("    WR = 90% → edge = +38.7% — but only true if BE formula is correct")
    print("  DIGITMATCH BE ≈ 10%: If digits uniform, WR = 10% → edge = 0%")
    print("  DIGITEVEN/ODD BE ≈ 51.3%: WR = 50% → edge = -1.3% (slight house edge)")


if __name__ == "__main__":
    asyncio.run(main())
