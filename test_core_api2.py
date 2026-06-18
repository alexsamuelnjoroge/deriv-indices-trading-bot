import urllib.request
import urllib.error
import json

PAT = "pat_6089f9c854db6fd085b6ddffdb92e977a9dcf6201a14d6cffe445161bf15d946"

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://developers.deriv.com",
    "Referer": "https://developers.deriv.com/playground/",
    "Connection": "keep-alive",
    "sec-ch-ua": '"Google Chrome";v="125", "Chromium";v="125"',
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
}

tests = [
    ("GET",  "https://auth.deriv.com/sessions/whoami"),
    ("GET",  "https://api-core.deriv.com/v1/derivatives/account"),
    ("POST", "https://api-core.deriv.com/v1/derivatives/otp"),
    ("POST", "https://api-core.deriv.com/v1/derivatives/session"),
]

for method, url in tests:
    try:
        hdrs = {**BROWSER_HEADERS, "Authorization": f"Bearer {PAT}"}
        req = urllib.request.Request(url, headers=hdrs, method=method)
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode()
            print(f"\nSUCCESS [{method}] {url}")
            print(f"  Response: {body[:500]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        print(f"HTTP {e.code} [{method}] {url}")
        print(f"  Body: {body}")
    except Exception as e:
        print(f"ERR [{method}] {url}: {type(e).__name__}: {str(e)[:100]}")
