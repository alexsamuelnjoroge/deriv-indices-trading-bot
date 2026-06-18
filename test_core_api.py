import urllib.request
import urllib.error
import json

PAT = "pat_6089f9c854db6fd085b6ddffdb92e977a9dcf6201a14d6cffe445161bf15d946"

tests = [
    ("GET",  "https://auth.deriv.com/sessions/whoami"),
    ("GET",  "https://api-core.deriv.com/v1/derivatives/account"),
    ("POST", "https://api-core.deriv.com/v1/derivatives/account"),
    ("GET",  "https://api-core.deriv.com/v1/derivatives/otp"),
    ("POST", "https://api-core.deriv.com/v1/derivatives/otp"),
    ("GET",  "https://api-core.deriv.com/v1/derivatives/session"),
    ("POST", "https://api-core.deriv.com/v1/derivatives/session"),
    ("GET",  "https://api-core.deriv.com/v1/auth"),
    ("POST", "https://api-core.deriv.com/v1/auth"),
]

for method, url in tests:
    for auth_header in [f"Bearer {PAT}", PAT]:
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": auth_header, "Content-Type": "application/json"},
                method=method
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read().decode()
                print(f"\nSUCCESS [{method}] {url}")
                print(f"  Auth: {auth_header[:30]}...")
                print(f"  Response: {body[:500]}")
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            print(f"HTTP {e.code} [{method}] {url} | Auth: {auth_header[:20]}...")
            if body:
                print(f"  Body: {body}")
            break
        except Exception as e:
            print(f"ERR [{method}] {url}: {type(e).__name__}: {str(e)[:100]}")
            break
