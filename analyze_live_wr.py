"""
Pull DIGITOVER/DIGITUNDER trade history from Deriv's statement API
and compute the actual live win rate per symbol.

This is the ground truth — independent of bot logs, race conditions,
and settlement field ambiguity.

Usage:  python analyze_live_wr.py
        python analyze_live_wr.py 500    # last 500 trades
"""
import asyncio
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from src.api.client import DerivClient

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    if not token:
        print("ERROR: DERIV_API_TOKEN not set")
        return

    client = DerivClient(token)
    await client.connect()

    # Fetch statement in pages
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

    # Filter to DIGITOVER/DIGITUNDER buy+sell pairs
    # Each trade has a buy (negative amount) and sell (positive amount) entry
    # Group by contract_id
    by_contract: dict = defaultdict(dict)
    for tx in all_transactions:
        cid = str(tx.get("contract_id", ""))
        action = tx.get("action_type", "")
        if action == "buy":
            by_contract[cid]["buy"] = tx
        elif action == "sell":
            by_contract[cid]["sell"] = tx

    # Analyze DIGITOVER/DIGITUNDER
    by_symbol: dict = defaultdict(lambda: {"wins": 0, "losses": 0, "profit": 0.0})

    for cid, pair in by_contract.items():
        sell = pair.get("sell", {})
        buy = pair.get("buy", {})
        desc = sell.get("longcode", "") or buy.get("longcode", "")

        is_digit = "digit" in desc.lower() and ("higher than" in desc.lower() or "lower than" in desc.lower())
        if not is_digit:
            continue

        symbol = sell.get("symbol") or buy.get("symbol", "unknown")
        amount = float(sell.get("amount", 0))
        stake = abs(float(buy.get("amount", 0)))
        profit = amount - stake

        if profit > 0:
            by_symbol[symbol]["wins"] += 1
        else:
            by_symbol[symbol]["losses"] += 1
        by_symbol[symbol]["profit"] += profit

    if not by_symbol:
        print("\nNo DIGITOVER/DIGITUNDER trades found in this range.")
        print("Tip: try a larger limit — python analyze_live_wr.py 2000")
        await client.disconnect()
        return

    print("\n--- DIGIT contract live results ---")
    for sym, stats in sorted(by_symbol.items()):
        w, l = stats["wins"], stats["losses"]
        total = w + l
        if total == 0:
            continue
        wr = w / total * 100
        expected_wr = 55.56
        gap = wr - expected_wr
        net = stats["profit"]
        print(
            f"\n{sym}: {w}W / {l}L / {total}T | "
            f"WR={wr:.1f}% (expected {expected_wr:.1f}%, gap={gap:+.1f}%) | "
            f"net P&L={net:+.2f}"
        )

    await client.disconnect()


asyncio.run(main())
