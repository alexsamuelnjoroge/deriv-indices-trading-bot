import asyncio, json, websockets

async def test():
    url = "wss://api.derivws.com/trading/v1/options/ws/demo?otp=X90MHtJK"
    print(f"Connecting to: {url}")
    try:
        async with websockets.connect(url) as ws:
            print("Connected!")
            # Try ping
            await ws.send(json.dumps({"ping": 1}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            print("Ping response:", json.dumps(resp, indent=2))
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

asyncio.run(test())
