import urllib.request

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

test_urls = [
    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT",
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT",
    "https://wwwn.cdc.gov/nchs/nhanes/2017-2018/DEMO_J.XPT",
    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.xpt",
]

for url in test_urls:
    try:
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=10)
        size = resp.headers.get("Content-Length", "unknown")
        print(f"OK  size={size}  {url}")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}  {url}")
    except Exception as e:
        print(f"ERR {e}  {url}")
