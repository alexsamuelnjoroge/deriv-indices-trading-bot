import asyncio, json, websockets

TOKEN = "pat_6089f9c854db6fd085b6ddffdb92e977a9dcf6201a14d6cffe445161bf15d946"
APP_IDS = [1089, 36544, 16929, 36300, 1411]

async def test(app_id):
    url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            await ws.send(json.dumps({"authorize": TOKEN, "req_id": 1}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            if "error" in resp:
                print(f"app_id={app_id}: ERROR - {resp['error']['code']} - {resp['error']['message']}")
            else:
                info = resp.get("authorize", {})
                print(f"app_id={app_id}: SUCCESS - loginid={info.get('loginid')} balance={info.get('balance')}")
    except Exception as e:
        print(f"app_id={app_id}: EXCEPTION - {type(e).__name__}: {e}")

async def main():
    for app_id in APP_IDS:
        await test(app_id)

asyncio.run(main())
