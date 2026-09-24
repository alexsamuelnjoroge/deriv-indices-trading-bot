"""
Pull DIGITOVER/DIGITUNDER trade history from Deriv's statement API
and compute the actual live win rate per symbol.

Usage:  python analyze_live_wr.py
        python analyze_live_wr.py 500    # last 500 trades
        python analyze_live_wr.py 100 dump   # also print first raw transaction
"""
import asyncio
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from src.api.client import DerivClient

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
DUMP_SAMPLE = len(sys.argv) > 2 and sys.argv[2] == "dump"

# R_ symbol names as they appear in contract longcodes
_LONGCODE_NAMES = {
    "Volatility 10 Index": "R_10",
    "Volatility 25 Index": "R_25",
    "Volatility 50 Index": "R_50",
    "Volatility 75 Index": "R_75",
    "Volatility 100 Index": "R_100",
    "Volatility 10 (1s) Index": "R_10_1S",
    "Volatility 25 (1s) Index": "R_25_1S",
    "Volatility 50 (1s) Index": "R_50_1S",
    "Volatility 75 (1s) Index": "R_75_1S",
    "Volatility 100 (1s) Index": "R_100_1S",
    "Crash 300 Index": "CRASH300",
    "Crash 500 Index": "CRASH500",
    "Crash 1000 Index": "CRASH1000",
    "Boom 300 Index": "BOOM300",
    "Boom 500 Index": "BOOM500",
    "Boom 1000 Index": "BOOM1000",
}


def _symbol_from_tx(tx: dict) -> str:
    """Extract underlying symbol from a statement transaction."""
    # Try direct fields first
    for field in ("symbol", "underlying_symbol", "underlying", "market"):
        v = tx.get(field)
        if v:
            return str(v)

    # Try shortcode: "DIGITOVER_R_50_S0P_1T_4_0" → second segment
    shortcode = tx.get("shortcode", "")
    if shortcode:
        parts = shortcode.split("_")
        if len(parts) >= 3:
            # Reconstruct symbol: could be R_50, R_75 etc.
            # Shortcode segments after contract type
            for i in range(1, len(parts) - 1):
                candidate = "_".join(parts[i:i+2])
                if candidate in ("R_10", "R_25", "R_50", "R_75", "R_100"):
                    return candidate

    # Fall back to longcode text match
    longcode = tx.get("longcode", "")
    for name, sym in _LONGCODE_NAMES.items():
        if name in longcode:
            return sym

    return "unknown"


async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    if not token:
        print("ERROR: DERIV_API_TOKEN not set")
        return

    client = DerivClient(token)
    await client.connect()

    all_transactions = []
    offset = 0
    page_size = 100

    print(f"Fetching up to {LIMIT} transactions...")
    while len(all_transactions) < LIMIT:
        fetch = min(page_size, LIMIT - len(all_transactions))
        resp = await client._send({
            "statement": 1,
            "description": 1,
            "limit": fetch,
            "offset": offset,
        })
        transactions = resp.get("statement", {}).get("transactions", [])
        if not transactions:
            break
        all_transactions.extend(transactions)
        offset += len(transactions)
        if len(transactions) < fetch:
            break

    print(f"Got {len(all_transactions)} transactions total")

    if DUMP_SAMPLE and all_transactions:
        print("\n--- SAMPLE RAW TRANSACTION (first entry) ---")
        print(dict(all_transactions[0]))
        print()

    # Group buy/sell by contract_id
    by_contract: dict = defaultdict(dict)
    for tx in all_transactions:
        cid = str(tx.get("contract_id", ""))
        action = tx.get("action_type", "")
        if action in ("buy", "sell"):
            by_contract[cid][action] = tx

    # Analyze DIGITOVER/DIGITUNDER only
    by_symbol: dict = defaultdict(lambda: {"wins": 0, "losses": 0, "profit": 0.0,
                                           "longcodes": set()})
    skipped = 0

    for cid, pair in by_contract.items():
        sell = pair.get("sell", {})
        buy  = pair.get("buy", {})
        tx   = sell or buy  # prefer sell (has outcome)
        desc = tx.get("longcode", "")

        is_digit = (
            "digit" in desc.lower()
            and ("higher than" in desc.lower() or "lower than" in desc.lower())
        )
        if not is_digit:
            skipped += 1
            continue

        symbol = _symbol_from_tx(tx) or _symbol_from_tx(buy)
        amount = float(sell.get("amount", 0))
        stake  = abs(float(buy.get("amount", 0)))
        profit = amount - stake

        by_symbol[symbol]["wins"]   += profit > 0
        by_symbol[symbol]["losses"] += profit <= 0
        by_symbol[symbol]["profit"] += profit
        by_symbol[symbol]["longcodes"].add(desc[:80])

    print(f"(skipped {skipped} non-digit transactions)")

    if not by_symbol:
        print("\nNo DIGITOVER/DIGITUNDER trades found.")
        print("Re-run with 'dump' flag to inspect raw fields:")
        print("  python analyze_live_wr.py 100 dump")
        await client.disconnect()
        return

    print("\n--- DIGIT contract live results ---")
    for sym, s in sorted(by_symbol.items()):
        w, l = s["wins"], s["losses"]
        total = w + l
        if total == 0:
            continue
        wr  = w / total * 100
        gap = wr - 55.56
        print(
            f"\n{sym}: {w}W / {l}L / {total}T | "
            f"WR={wr:.1f}% (expected 55.6%, gap={gap:+.1f}%) | "
            f"net={s['profit']:+.2f}"
        )
        # Show sample longcodes to confirm filter is catching the right contracts
        for lc in list(s["longcodes"])[:2]:
            print(f"  eg: {lc}")

    await client.disconnect()


asyncio.run(main())
