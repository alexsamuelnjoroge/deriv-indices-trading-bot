"""
Probe exotic contract types on Volatility indices with correct API parameters.

Covers:
  - DIGITDIFF/DIGITMATCH/DIGITOVER/DIGITUNDER (correct last_digit param)
  - ONETOUCH/NOTOUCH (correct barrier sizing per symbol)
  - TURBOSLONG/TURBOSSHORT (knock-out leveraged)
  - VANILLALONGCALL/PUT (standard options)
  - TICKHIGH/TICKLOW, RESETCALL/RESETPUT, ASIANU/ASIAND
  - RUNHIGH/RUNLOW, EXPIRY*, UPORDOWN, RANGE

Usage:
  python check_exotic_contracts.py 2>/dev/null
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, ".")
from src.api.client import DerivClient

SYMBOLS   = ["R_10", "R_100", "1HZ10V", "JD10"]
STAKE     = 1.0
SEP       = "=" * 78
THIN      = "-" * 78


async def probe(client, req: dict) -> dict:
    try:
        resp = await client._send(req)
        p   = resp.get("proposal", {})
        err = resp.get("error", {})
        if err:
            return {"ok": False, "code": err.get("code", "?"),
                    "msg": err.get("message", "?")[:60]}
        payout = float(p.get("payout", 0) or 0)
        ask    = float(p.get("ask_price", STAKE) or STAKE)
        profit = (payout - ask) / ask * 100 if ask > 0 else 0
        be     = 100 / (1 + profit / 100) if profit > 0 else 0
        extra  = {}
        # Grab any barrier or spot info
        for k in ("spot", "barrier", "high_barrier", "low_barrier",
                  "multiplier", "strike", "contract_details"):
            if k in p:
                extra[k] = p[k]
        return {"ok": True, "payout": payout, "ask": ask,
                "profit_pct": profit, "be": be, "extra": extra}
    except Exception as e:
        return {"ok": False, "code": "EXCEPTION", "msg": str(e)[:80]}


def print_result(label: str, r: dict):
    if r["ok"]:
        extra_str = ""
        if r["extra"].get("barrier"):
            extra_str = f"  barrier={r['extra']['barrier']}"
        elif r["extra"].get("strike"):
            extra_str = f"  strike={r['extra']['strike']}"
        print(f"  {label:<40}  payout={r['payout']:.4f}  "
              f"profit%={r['profit_pct']:.1f}%  BE={r['be']:.1f}%{extra_str}")
    else:
        print(f"  {label:<40}  ERROR [{r['code']}]: {r['msg']}")


async def main():
    token  = os.getenv("DERIV_TOKEN") or os.getenv("DERIV_API_TOKEN", "")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    client = DerivClient(api_token=token, app_id=app_id)
    await client.connect()

    base = {"proposal": 1, "amount": STAKE, "basis": "stake", "currency": "USD"}

    for symbol in SYMBOLS:
        print()
        print(SEP)
        print(f"  {symbol}")
        print(SEP)

        # ── Get current spot price for barrier sizing ──────────────────
        spot_resp = await client._send({"ticks": symbol})
        spot = float(spot_resp.get("tick", {}).get("quote", 100))
        print(f"  Current spot: {spot}")
        print(THIN)

        # Digit contracts — try several candidate parameter names
        print("  DIGIT contracts (probing correct param name):")
        for ct in ["DIGITDIFF", "DIGITMATCH"]:
            found = False
            for param_name, param_val in [
                ("barrier", "5"),
                ("barrier", 5),
                ("prediction", 5),
                ("digit", 5),
                ("selected_tick", 5),
            ]:
                r = await probe(client, {**base, "underlying_symbol": symbol,
                    "contract_type": ct, "duration": 1, "duration_unit": "t",
                    **{param_name: param_val}})
                status = "OK" if r["ok"] else r.get("code", "ERR")
                msg    = r.get("msg", "")[:50] if not r["ok"] else ""
                print(f"    {ct} param={param_name}={param_val!r}  → {status}  {msg}")
                if r["ok"]:
                    found = True
                    break
                if r.get("code") not in ("InputValidationFailed", "ContractCreationFailure"):
                    break   # unexpected error, stop trying

        for ct in ["DIGITOVER", "DIGITUNDER", "DIGITEVEN", "DIGITODD"]:
            for param_name, param_val in [
                ("barrier", "5"),
                ("barrier", 5),
                (None, None),     # no digit param (for EVEN/ODD)
            ]:
                req_extra = {} if param_name is None else {param_name: param_val}
                r = await probe(client, {**base, "underlying_symbol": symbol,
                    "contract_type": ct, "duration": 1, "duration_unit": "t",
                    **req_extra})
                status = "OK" if r["ok"] else r.get("code", "ERR")
                msg    = r.get("msg", "")[:50] if not r["ok"] else ""
                label  = f"no-param" if param_name is None else f"{param_name}={param_val!r}"
                print(f"    {ct} {label}  → {status}  {msg}")
                if r["ok"] or param_name is None:
                    break

        print()

        # Touch/No Touch — try barrier as absolute price offset
        print("  TOUCH/NO-TOUCH (absolute barrier offsets):")
        for ct in ["ONETOUCH", "NOTOUCH"]:
            for pct_offset in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
                barrier = f"+{spot * pct_offset:.4f}"
                r = await probe(client, {**base, "underlying_symbol": symbol,
                    "contract_type": ct, "duration": 10, "duration_unit": "t",
                    "barrier": barrier})
                if r["ok"]:
                    print_result(f"  {ct} barrier={barrier} (10t)", r)
                    break
                elif "Barrier" not in r.get("msg", "") and "barrier" not in r.get("msg", "").lower():
                    print_result(f"  {ct} barrier={barrier} (10t)", r)
                    break
            # Also try different durations
            for dur in [5, 10, 20, 100]:
                barrier = f"+{spot * 0.01:.4f}"
                r = await probe(client, {**base, "underlying_symbol": symbol,
                    "contract_type": ct, "duration": dur, "duration_unit": "t",
                    "barrier": barrier})
                if r["ok"]:
                    print_result(f"  {ct} barrier=1% ({dur}t)", r)
                    break

        print()

        # Turbos
        print("  TURBOS (knock-out leveraged):")
        for ct in ["TURBOSLONG", "TURBOSSHORT"]:
            # Barrier below spot for LONG, above for SHORT
            for pct in [0.02, 0.05, 0.10]:
                if ct == "TURBOSLONG":
                    barrier = f"{spot * (1 - pct):.4f}"
                else:
                    barrier = f"{spot * (1 + pct):.4f}"
                r = await probe(client, {**base, "underlying_symbol": symbol,
                    "contract_type": ct, "duration": 10, "duration_unit": "t",
                    "barrier": barrier})
                if r["ok"]:
                    print_result(f"  {ct} barrier={pct*100:.0f}% from spot ({barrier})", r)
                    break
                elif "barrier" not in r.get("msg", "").lower() and "Barrier" not in r.get("msg", ""):
                    print_result(f"  {ct} barrier={pct*100:.0f}% ({barrier})", r)
                    break

        print()

        # Vanillas
        print("  VANILLAS (standard options):")
        for ct in ["VANILLALONGCALL", "VANILLALONGPUT"]:
            for dur in [60, 300, 900]:  # 1min, 5min, 15min
                r = await probe(client, {**base, "underlying_symbol": symbol,
                    "contract_type": ct, "duration": dur, "duration_unit": "s"})
                if r["ok"] or "Duration" not in r.get("msg", ""):
                    print_result(f"  {ct} {dur}s", r)
                    break

        print()

        # Tick contracts
        print("  TICK contracts (highest/lowest over N ticks):")
        for ct in ["TICKHIGH", "TICKLOW"]:
            r = await probe(client, {**base, "underlying_symbol": symbol,
                "contract_type": ct, "duration": 5, "duration_unit": "t"})
            print_result(f"  {ct} 5t", r)

        print()

        # Asian options
        print("  ASIAN (average price options):")
        for ct in ["ASIANU", "ASIAND"]:
            r = await probe(client, {**base, "underlying_symbol": symbol,
                "contract_type": ct, "duration": 5, "duration_unit": "t"})
            print_result(f"  {ct} 5t", r)

        print()

        # Reset options
        print("  RESET options:")
        for ct in ["RESETCALL", "RESETPUT"]:
            r = await probe(client, {**base, "underlying_symbol": symbol,
                "contract_type": ct, "duration": 5, "duration_unit": "t"})
            print_result(f"  {ct} 5t", r)

        print()

        # Run contracts
        print("  RUN contracts (consecutive ticks):")
        for ct in ["RUNHIGH", "RUNLOW"]:
            for dur in [3, 5]:
                r = await probe(client, {**base, "underlying_symbol": symbol,
                    "contract_type": ct, "duration": dur, "duration_unit": "t"})
                if r["ok"]:
                    print_result(f"  {ct} {dur}t", r)
                    break
                elif "Duration" not in r.get("msg", ""):
                    print_result(f"  {ct} {dur}t", r)
                    break

        print()

        # Expiry range
        print("  EXPIRY RANGE / MISS:")
        for ct in ["EXPIRYRANGE", "EXPIRYMISS"]:
            for dur_unit, dur in [("d", 1), ("h", 1), ("m", 5)]:
                barrier = f"+{spot * 0.01:.4f}"
                r = await probe(client, {**base, "underlying_symbol": symbol,
                    "contract_type": ct, "duration": dur, "duration_unit": dur_unit,
                    "barrier": barrier, "barrier2": f"-{spot * 0.01:.4f}"})
                if r["ok"]:
                    print_result(f"  {ct} {dur}{dur_unit}", r)
                    break
                elif "Duration" not in r.get("msg", ""):
                    print_result(f"  {ct} {dur}{dur_unit}", r)
                    break

    print()
    await client.disconnect()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
