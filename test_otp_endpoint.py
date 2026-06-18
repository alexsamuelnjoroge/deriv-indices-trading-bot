import urllib.request
import urllib.error
import json

PAT = "pat_6089f9c854db6fd085b6ddffdb92e977a9dcf6201a14d6cffe445161bf15d946"

endpoints = [
    "https://api.derivws.com/trading/v1/auth",
    "https://api.derivws.com/trading/v1/auth/otp",
    "https://api.derivws.com/trading/v1/otp",
    "https://api.derivws.com/trading/v1/session",
    "https://api.derivws.com/trading/v1/tokens",
    "https://api.derivws.com/auth",
    "https://api.derivws.com/v1/auth",
]

headers_variants = [
    {"Authorization": f"Bearer {PAT}"},
    {"Authorization": PAT},
    {"X-PAT-Token": PAT},
    {"pat-token": PAT},
]

for url in endpoints:
    for hdrs in headers_variants[:1]:  # test first header variant per URL
        try:
            req = urllib.request.Request(url, headers=hdrs, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
                print(f"SUCCESS {url}: {body[:200]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:100]
            print(f"HTTP {e.code} {url}: {body}")
        except Exception as e:
            print(f"ERR {url}: {type(e).__name__}: {str(e)[:80]}")
