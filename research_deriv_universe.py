"""
Phase 1: Deriv synthetic universe discovery.

Queries the Deriv API to enumerate:
  - All available synthetic index symbols
  - All contract types available per symbol
  - Payout percentages per contract type
  - Minimum/maximum durations

Outputs a structured report grouped by symbol category and contract type,
highlighting combinations worth backtesting in Phase 2.

Usage:
    python research_deriv_universe.py
    python research_deriv_universe.py --category volatility
    python research_deriv_universe.py --category crash_boom
    python research_deriv_universe.py --category jump
    python research_deriv_universe.py --category step
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict

from loguru import logger

# Use the existing Deriv client
sys.path.insert(0, os.path.dirname(__file__))
from src.api.client import DerivClient

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")

# ── Symbol categories ─────────────────────────────────────────────────────────
CATEGORIES = {
    "volatility":  ["R_10", "R_25", "R_50", "R_75", "R_100",
                    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
                    "1HZ150V", "1HZ200V", "1HZ250V", "1HZ300V"],
    "crash_boom":  ["BOOM50", "BOOM100", "BOOM150N", "BOOM300N", "BOOM500",
                    "BOOM600", "BOOM900", "BOOM1000",
                    "CRASH50", "CRASH100", "CRASH150N", "CRASH300N", "CRASH500",
                    "CRASH600", "CRASH900", "CRASH1000"],
    "step":        ["STPIDX"],
    "jump":        ["JD10", "JD25", "JD50", "JD75", "JD100"],
    "range_break": ["RB100", "RB200"],
    "dex":         ["DEX600DN", "DEX600UP", "DEX900DN", "DEX900UP",
                    "DEX1500DN", "DEX1500UP"],
}

ALL_SYMBOLS = [s for syms in CATEGORIES.values() for s in syms]

# ── Contract types to check ───────────────────────────────────────────────────
CONTRACT_TYPES = [
    "CALL",        # Rise/Fall
    "PUT",
    "ONETOUCH",    # Touch / No Touch
    "NOTOUCH",
    "RANGE",       # In / Out
    "UPORDOWN",
    "ASIANU",      # Asian Up/Down
    "ASIAND",
    "DIGITOVER",   # Digits
    "DIGITUNDER",
    "DIGITEVEN",
    "DIGITODD",
    "DIGITMATCH",
    "DIGITDIFF",
    "ACCU",        # Accumulator
    "MULTUP",      # Multiplier
    "MULTDOWN",
    "TICKHIGH",    # Lookback
    "TICKLOW",
    "RESETCALL",   # Reset
    "RESETPUT",
    "RUNHIGH",     # Run High/Low
    "RUNLOW",
]

# Contract type friendly names + breakeven calculation notes
CONTRACT_META = {
    "CALL":        {"group": "Rise/Fall",    "direction": True,   "payout_type": "fixed"},
    "PUT":         {"group": "Rise/Fall",    "direction": True,   "payout_type": "fixed"},
    "ONETOUCH":    {"group": "Touch",        "direction": False,  "payout_type": "fixed"},
    "NOTOUCH":     {"group": "Touch",        "direction": False,  "payout_type": "fixed"},
    "RANGE":       {"group": "Range",        "direction": False,  "payout_type": "fixed"},
    "UPORDOWN":    {"group": "Range",        "direction": False,  "payout_type": "fixed"},
    "ASIANU":      {"group": "Asian",        "direction": True,   "payout_type": "fixed"},
    "ASIAND":      {"group": "Asian",        "direction": True,   "payout_type": "fixed"},
    "DIGITOVER":   {"group": "Digits",       "direction": False,  "payout_type": "fixed"},
    "DIGITUNDER":  {"group": "Digits",       "direction": False,  "payout_type": "fixed"},
    "DIGITEVEN":   {"group": "Digits",       "direction": False,  "payout_type": "fixed"},
    "DIGITODD":    {"group": "Digits",       "direction": False,  "payout_type": "fixed"},
    "DIGITMATCH":  {"group": "Digits",       "direction": False,  "payout_type": "fixed"},
    "DIGITDIFF":   {"group": "Digits",       "direction": False,  "payout_type": "fixed"},
    "ACCU":        {"group": "Accumulator",  "direction": False,  "payout_type": "variable"},
    "MULTUP":      {"group": "Multiplier",   "direction": True,   "payout_type": "variable"},
    "MULTDOWN":    {"group": "Multiplier",   "direction": True,   "payout_type": "variable"},
    "TICKHIGH":    {"group": "Lookback",     "direction": False,  "payout_type": "variable"},
    "TICKLOW":     {"group": "Lookback",     "direction": False,  "payout_type": "variable"},
    "RESETCALL":   {"group": "Reset",        "direction": True,   "payout_type": "fixed"},
    "RESETPUT":    {"group": "Reset",        "direction": True,   "payout_type": "fixed"},
    "RUNHIGH":     {"group": "Run",          "direction": False,  "payout_type": "variable"},
    "RUNLOW":      {"group": "Run",          "direction": False,  "payout_type": "variable"},
}


async def get_contracts_for(client: DerivClient, symbol: str) -> dict:
    """Query available contracts for a symbol."""
    try:
        resp = await client._send({"contracts_for": symbol})
        return resp.get("contracts_for", {})
    except Exception as e:
        return {"error": str(e)}


async def run(category_filter: str | None):
    api_token = os.getenv("DERIV_API_TOKEN", "")
    app_id    = int(os.getenv("DERIV_APP_ID", "1089"))

    client = DerivClient(api_token=api_token, app_id=app_id)
    await client.connect()

    symbols_to_check = []
    for cat, syms in CATEGORIES.items():
        if category_filter and cat != category_filter:
            continue
        symbols_to_check.extend([(cat, s) for s in syms])

    # Results: symbol → list of {contract_type, min_dur, max_dur, payout}
    results: dict[str, list] = {}
    unavailable = []

    print(f"\nQuerying {len(symbols_to_check)} symbols...\n")

    for cat, symbol in symbols_to_check:
        data = await get_contracts_for(client, symbol)
        if "error" in data or not data.get("available"):
            unavailable.append(symbol)
            continue

        contracts = []
        for c in data["available"]:
            ct = c.get("contract_type", "")
            if ct not in CONTRACT_TYPES:
                continue
            entry = {
                "contract_type":    ct,
                "group":            CONTRACT_META.get(ct, {}).get("group", ct),
                "min_duration":     c.get("min_contract_duration", "?"),
                "max_duration":     c.get("max_contract_duration", "?"),
                "payout":           c.get("payout_limit", None),
                "barriers":         c.get("barriers", 0),
                "expiry_type":      c.get("expiry_type", "?"),
                "start_type":       c.get("start_type", "?"),
            }
            # For fixed-payout contracts: extract payout% if available
            if "payout_limit" in c and c["payout_limit"]:
                entry["payout_pct"] = round((c["payout_limit"] - 1) * 100, 1)
            contracts.append(entry)

        if contracts:
            results[f"{cat}:{symbol}"] = contracts

    await client.disconnect()

    # ── Print report ──────────────────────────────────────────────────────────
    print("=" * 100)
    print("  DERIV SYNTHETIC UNIVERSE — Available Contracts")
    print("=" * 100)

    # Group by category
    by_category = defaultdict(list)
    for key, contracts in results.items():
        cat, sym = key.split(":", 1)
        by_category[cat].append((sym, contracts))

    for cat, sym_list in by_category.items():
        print(f"\n{'─'*100}")
        print(f"  [{cat.upper()}]")
        print(f"{'─'*100}")

        # Show which groups are available per symbol
        for symbol, contracts in sorted(sym_list):
            groups = defaultdict(list)
            for c in contracts:
                groups[c["group"]].append(c)

            print(f"\n  {symbol}")
            for group, items in sorted(groups.items()):
                cts = ", ".join(sorted(set(i["contract_type"] for i in items)))
                durations = set(f"{i['min_duration']}–{i['max_duration']}" for i in items)
                dur_str = " / ".join(sorted(durations))
                payouts = [i.get("payout_pct") for i in items if i.get("payout_pct")]
                pay_str = f"  payout={min(payouts):.0f}–{max(payouts):.0f}%" if payouts else ""
                print(f"    {group:<15} {cts:<35} dur:{dur_str:<20}{pay_str}")

    # ── Summary: best opportunities ───────────────────────────────────────────
    print()
    print("=" * 100)
    print("  STRATEGIC OPPORTUNITIES — Contracts with exploitable structure")
    print("=" * 100)

    opportunities = []

    for key, contracts in results.items():
        cat, symbol = key.split(":", 1)
        for c in contracts:
            group = c["group"]
            ct    = c["contract_type"]
            pay   = c.get("payout_pct", None)

            # Flag interesting combinations
            reason = None

            if group == "Accumulator" and cat == "crash_boom":
                reason = "Post-spike ACCU — structural edge if spike interval is long"
            elif group == "Digits" and ct in ("DIGITOVER", "DIGITUNDER"):
                reason = "Digit frequency — test for non-uniform last-digit distribution"
            elif group == "Lookback":
                reason = "Lookback — always pays optimal entry, but payout must exceed 1x stake"
            elif group == "Asian":
                reason = "Asian — avg vs entry, smoother than binary; less noise"
            elif group == "Run":
                reason = "Run High/Low — each consecutive correct tick earns; very sensitive to edge size"
            elif group == "Reset":
                reason = "Reset — barrier resets mid-contract; unique risk profile"
            elif group == "Touch" and cat == "volatility":
                reason = "No Touch on volatility — ATR-based barrier prediction"

            if reason:
                opportunities.append({
                    "symbol": symbol, "category": cat, "group": group,
                    "contract_type": ct, "payout_pct": pay, "reason": reason,
                    "min_dur": c["min_duration"], "max_dur": c["max_duration"],
                })

    # Print opportunities grouped by contract group
    opp_by_group = defaultdict(list)
    for o in opportunities:
        opp_by_group[o["group"]].append(o)

    for group, opps in sorted(opp_by_group.items()):
        print(f"\n  ── {group} ──")
        for o in opps:
            pay = f"  payout={o['payout_pct']:.0f}%" if o["payout_pct"] else ""
            print(f"    {o['symbol']:<15} {o['contract_type']:<15} "
                  f"dur:{o['min_dur']}–{o['max_dur']:<10}{pay}")
            print(f"      → {o['reason']}")

    if unavailable:
        print(f"\n  Unavailable/error symbols: {', '.join(unavailable)}")

    # Save raw results for Phase 2
    out_path = "data/deriv_universe.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "unavailable": unavailable}, f, indent=2)
    print(f"\n  Raw data saved to {out_path}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None,
                        choices=list(CATEGORIES.keys()),
                        help="Filter to one category")
    args = parser.parse_args()
    asyncio.run(run(args.category))


if __name__ == "__main__":
    main()
