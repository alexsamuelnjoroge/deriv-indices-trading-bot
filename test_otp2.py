import urllib.request
import urllib.error
import json

PAT = "pat_6089f9c854db6fd085b6ddffdb92e977a9dcf6201a14d6cffe445161bf15d946"

tests = [
    # (url, method, headers)
    ("https://api.derivws.com/trading/v1/options/token", "POST", {"Authorization": f"Bearer {PAT}"}),
    ("https://api.derivws.com/trading/v1/options/token", "GET",  {"Authorization": f"Bearer {PAT}"}),
    ("https://api.derivws.com/trading/v1/options/auth",  "POST", {"Authorization": f"Bearer {PAT}"}),
    ("https://api.derivws.com/trading/v1/options/otp",   "POST", {"Authorization": f"Bearer {PAT}"}),
    ("https://api.deriv.com/trading/v1/options/otp",     "POST", {"Authorization": f"Bearer {PAT}"}),
    ("https://api.deriv.com/v1/auth",                    "POST", {"Authorization": f"Bearer {PAT}"}),
    ("https://oauth.deriv.com/oauth2/token",             "POST", {"Authorization": f"Bearer {PAT}"}),
    # try PAT as a query param
    (f"https://api.derivws.com/trading/v1/options/ws/demo?otp={PAT}", "GET", {}),
]

for url, method, hdrs in tests:
    try:
        req = urllib.request.Request(url, headers=hdrs, method=method)
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode()
            print(f"SUCCESS [{method}] {url}")
            print(f"  Response: {body[:300]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:150]
        print(f"HTTP {e.code} [{method}] {url}: {body}")
    except Exception as e:
        print(f"ERR [{method}] {url}: {type(e).__name__}: {str(e)[:80]}")
