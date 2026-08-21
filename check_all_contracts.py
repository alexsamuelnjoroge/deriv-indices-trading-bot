"""
Comprehensive contract type survey across all synthetic index categories.

For each symbol, shows every available contract type with payout,
duration options, and stake limits. Identifies gaps and opportunities.

Usage:
  python check_all_contracts.py 2>/dev/null
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient

# Representative sample — one symbol per category
SURVEY_SYMBOLS = [
    # CRASH/BOOM
    "BOOM1000", "BOOM600", "CRASH1000", "CRASH600",
    # Standard Volatility
    "R_10", "R_25", "R_50", "R_75", "R_100",
    # 1Hz Volatility
    "1HZ10V", "1HZ25V", "1HZ100V",
    # Jump
    "JD10", "JD25",
]

STAKE = 1.0

# Contract types to probe for payout
PROBE_TYPES = {
    # Directional
    "CALL":       {"duration": 5,  "duration_unit": "t"},
    "PUT":        {"duration": 5,  "duration_unit": "t"},
    "ONETOUCH":   {"duration": 5,  "duration_unit": "t"},
    "NOTOUCH":    {"duration": 5,  "duration_unit": "t"},
    "EXPIRYRANGE":{"duration": 5,  "duration_unit": "t"},
    "EXPIRYMISS": {"duration": 5,  "duration_unit": "t"},
    "RANGE":      {"duration": 5,  "duration_unit": "t"},
    "UPORDOWN":   {"duration": 5,  "duration_unit": "t"},
    # Digit
    "DIGITDIFF":  {"duration": 1,  "duration_unit": "t"},
    "DIGITMATCH": {"duration": 1,  "duration_unit": "t"},
    "DIGITOVER":  {"duration": 1,  "duration_unit": "t"},
    "DIGITUNDER": {"duration": 1,  "duration_unit": "t"},
    "DIGITEVEN":  {"duration": 1,  "duration_unit": "t"},
    "DIGITODD":   {"duration": 1,  "duration_unit": "t"},
    # Growth
    "ACCU":       None,   # special handling — no duration
    "MULTUP":     None,   # special handling — multiplier
    "TURBOCALL":  None,   # special handling — barrier
    "TURBOPUTS":  None,
    "VANILLALONGCALL": None,
    "VANILLALONGPUT":  None,
}

SEP  = "=" * 80
THIN = "-" * 80


async def get_contracts_for(client, symbol: str) -> set:
    """Return set of available contract type names for a symbol."""
    resp = await client._send({"contracts_for": symbol})
    available = resp.get("contracts_for", {}).get("available", [])
    return {c.get("contract_type") for c in available}


async def probe_payout(client, symbol: str, ct: str, cfg: dict) -> dict | None:
    """Attempt a proposal for a contract type. Returns payout info or None."""
    req = {
        "proposal": 1,
        "amount":   STAKE,
        "basis":    "stake",
        "contract_type": ct,
        "currency": "USD",
        "underlying_symbol": symbol,
    }
    req.update(cfg)

    # Digit contracts need extra params
    if ct in ("DIGITDIFF", "DIGITMATCH"):
        req["selected_tick"] = 5
    elif ct in ("DIGITOVER", "DIGITUNDER"):
        req["selected_tick"] = 5
    # Touch/No Touch need a barrier
    elif ct in ("ONETOUCH", "NOTOUCH"):
        req["duration"] = 5
        req["duration_unit"] = "t"
        req["barrier"] = "+0.001"   # 0.1% above/below spot
    elif ct in ("EXPIRYRANGE", "EXPIRYMISS", "RANGE", "UPORDOWN"):
        req["duration"] = 5
        req["duration_unit"] = "t"
        req["barrier"] = "+0.001"
        req["barrier2"] = "-0.001"

    try:
        resp = await client._send(req)
        p   = resp.get("proposal", {})
        err = resp.get("error", {})
        if err:
            return {"error": err.get("code", err.get("message", "?"))}
        payout = float(p.get("payout", 0) or 0)
        ask    = float(p.get("ask_price", STAKE) or STAKE)
        profit_pct = (payout - ask) / ask * 100 if ask > 0 else 0
        return {"payout": payout, "ask": ask, "profit_pct": profit_pct}
    except Exception as e:
        return {"error": str(e)[:60]}


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    # ── Step 1: Availability matrix ──────────────────────────────────────
    print("\n" + SEP)
    print("  CONTRACT TYPE AVAILABILITY MATRIX")
    print(SEP)

    availability: dict[str, set] = {}
    for symbol in SURVEY_SYMBOLS:
        try:
            types = await get_contracts_for(client, symbol)
            availability[symbol] = types
        except Exception as e:
            availability[symbol] = set()
            print(f"  ERROR fetching {symbol}: {e}")

    # Group contract types seen anywhere
    all_types = sorted({ct for types in availability.values() for ct in types if ct})

    # Print matrix
    col_w = 10
    header = f"  {'Contract':<22}" + "".join(f"{s[:col_w]:>{col_w}}" for s in SURVEY_SYMBOLS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ct in all_types:
        row = f"  {ct:<22}"
        for sym in SURVEY_SYMBOLS:
            mark = "Y" if ct in availability.get(sym, set()) else "."
            row += f"{mark:>{col_w}}"
        print(row)

    # ── Step 2: Payout survey for key symbols ────────────────────────────
    PAYOUT_SYMBOLS = ["BOOM1000", "R_10", "R_100", "1HZ10V", "JD10"]

    print()
    print(SEP)
    print("  PAYOUT SURVEY  (stake=$1.00, 5-tick duration where applicable)")
    print(SEP)

    for symbol in PAYOUT_SYMBOLS:
        avail = availability.get(symbol, set())
        print(f"\n  {symbol}  |  available: {', '.join(sorted(avail))}")
        print(THIN)
        print(f"  {'Contract':<20}  {'Payout':>8}  {'Profit%':>8}  Note")
        print(f"  {'-'*20}  {'-'*8}  {'-'*8}  ----")

        for ct, cfg in PROBE_TYPES.items():
            if ct not in avail:
                continue
            if cfg is None:
                # Skip complex types for now — handled separately
                print(f"  {ct:<20}  {'(see below)':>8}")
                continue
            r = await probe_payout(client, symbol, ct, cfg)
            if r is None:
                continue
            if "error" in r:
                print(f"  {ct:<20}  {'N/A':>8}  {'N/A':>8}  {r['error']}")
            else:
                be = 100 / (1 + r["profit_pct"] / 100) if r["profit_pct"] > 0 else 0
                print(f"  {ct:<20}  {r['payout']:>8.4f}  {r['profit_pct']:>7.1f}%  BE={be:.1f}%")

    # ── Step 3: Digit distribution snapshot (Volatility) ─────────────────
    print()
    print(SEP)
    print("  DIGIT CONTRACT PAYOUT DETAILS")
    print(SEP)

    for symbol in ["R_10", "R_25", "R_50", "R_75", "R_100", "1HZ10V"]:
        if "DIGITDIFF" not in availability.get(symbol, set()):
            continue
        print(f"\n  {symbol}")
        for ct, digit in [("DIGITDIFF", 5), ("DIGITMATCH", 5),
                          ("DIGITOVER", 4), ("DIGITUNDER", 5),
                          ("DIGITEVEN", None), ("DIGITODD", None)]:
            if ct not in availability.get(symbol, set()):
                continue
            req = {
                "proposal": 1, "amount": STAKE, "basis": "stake",
                "contract_type": ct, "currency": "USD",
                "duration": 1, "duration_unit": "t",
                "underlying_symbol": symbol,
            }
            if digit is not None:
                req["selected_tick"] = digit
            try:
                resp = await client._send(req)
                p   = resp.get("proposal", {})
                err = resp.get("error", {})
                if err:
                    print(f"    {ct:<14} digit={digit}  ERROR: {err.get('code','?')}")
                    continue
                payout = float(p.get("payout", 0) or 0)
                ask    = float(p.get("ask_price", STAKE) or STAKE)
                profit_pct = (payout - ask) / ask * 100 if ask > 0 else 0
                print(f"    {ct:<14} digit={digit}  payout={payout:.4f}  profit%={profit_pct:.1f}%")
            except Exception as e:
                print(f"    {ct:<14} ERROR: {e}")

    # ── Step 4: Touch/No Touch on BOOM/CRASH ─────────────────────────────
    print()
    print(SEP)
    print("  TOUCH/NO-TOUCH — BOOM/CRASH (barrier survey at multiple offsets)")
    print(SEP)

    for symbol in ["BOOM1000", "CRASH1000", "BOOM600", "CRASH600"]:
        avail = availability.get(symbol, set())
        if "ONETOUCH" not in avail and "NOTOUCH" not in avail:
            print(f"  {symbol}: Touch/No Touch NOT available")
            continue

        print(f"\n  {symbol}:")
        for ct in ["ONETOUCH", "NOTOUCH"]:
            if ct not in avail:
                continue
            for dur in [5, 10, 20, 50]:
                req = {
                    "proposal": 1, "amount": STAKE, "basis": "stake",
                    "contract_type": ct, "currency": "USD",
                    "duration": dur, "duration_unit": "t",
                    "underlying_symbol": symbol,
                    "barrier": "+0.001",
                }
                try:
                    resp = await client._send(req)
                    p   = resp.get("proposal", {})
                    err = resp.get("error", {})
                    if err:
                        print(f"    {ct} {dur}t  ERROR: {err.get('code','?')}")
                        break
                    payout = float(p.get("payout", 0) or 0)
                    ask    = float(p.get("ask_price", STAKE) or STAKE)
                    profit_pct = (payout - ask) / ask * 100 if ask > 0 else 0
                    print(f"    {ct} {dur:>3}t  payout={payout:.4f}  profit%={profit_pct:.1f}%")
                except Exception as e:
                    print(f"    {ct} {dur}t  ERROR: {e}")

    print()
    await client.disconnect()
    print("Survey complete.")


if __name__ == "__main__":
    asyncio.run(main())
