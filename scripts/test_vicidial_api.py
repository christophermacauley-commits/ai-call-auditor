import os
import urllib.parse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

base = os.getenv("VICIDIAL_API_URL")
user = os.getenv("VICIDIAL_API_USER")
password = os.getenv("VICIDIAL_API_PASS")

if not base or not user or not password:
    raise SystemExit("Missing VICIDIAL_API_URL, VICIDIAL_API_USER, or VICIDIAL_API_PASS in .env")

params = {
    "source": "AIAUDITOR",
    "user": user,
    "pass": password,
    "function": "version",
}

url = base + "?" + urllib.parse.urlencode(params)

with urllib.request.urlopen(url, timeout=30) as response:
    body = response.read().decode("utf-8", errors="replace").strip()

print(body)
