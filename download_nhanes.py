"""
NHANES Data Downloader - CORRECT URLs
Real URL format: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}.XPT
"""
import sys, urllib.request, urllib.error, pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/raw/nhanes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# year_start -> (letter, year_label)
CYCLES = {
    "2013": ("H", "2013-2014"),
    "2015": ("I", "2015-2016"),
    "2017": ("J", "2017-2018"),
}

FILES = {
    "DEMO":   "Demographics (age, sex, ethnicity)",
    "CBC":    "Complete Blood Count",
    "BIOPRO": "Biochemistry Panel (albumin, ALT, AST, ALP, bilirubin, creatinine)",
    "HSCRP":  "High-Sensitivity CRP",
    "FERTIN": "Ferritin",
    "MCQ":    "Medical Conditions (cancer MCQ220/MCQ230)",
}

# CORRECT URL format (verified working - returns 3.4MB real data)
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{prefix}_{letter}.XPT"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print("=" * 65)
print("NHANES DOWNLOADER (Correct URLs) - Cancer Early Detection Study")
print("=" * 65)

downloaded, failed = [], []

for year_start, (letter, year_label) in CYCLES.items():
    print(f"\nCycle: {year_label}")
    for prefix, desc in FILES.items():
        url      = BASE_URL.format(year=year_start, prefix=prefix, letter=letter)
        filename = f"{prefix}_{letter}_{year_label.replace('-','_')}.XPT"
        out_path = OUTPUT_DIR / filename

        # Skip if already properly downloaded (real file > 100KB)
        if out_path.exists() and out_path.stat().st_size > 100_000:
            print(f"  [OK]  {filename} ({out_path.stat().st_size/1024:.0f} KB) already exists")
            downloaded.append(str(out_path))
            continue

        try:
            print(f"  [GET] {prefix} ({desc})...", end=" ", flush=True)
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            size_kb = len(data) / 1024
            if size_kb < 50:
                print(f"WARNING - only {size_kb:.0f} KB, might be error page")
                failed.append((filename, "too small"))
            else:
                print(f"{size_kb:.0f} KB - OK")
                downloaded.append(str(out_path))
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((filename, str(e)))

print("\n" + "=" * 65)
print(f"Downloaded : {len(downloaded)}")
print(f"Failed     : {len(failed)}")
if failed:
    for f, e in failed:
        print(f"  FAIL: {f} -> {e}")

print("\nPREVIEW")
print("=" * 65)

grand_n, grand_cancer = 0, 0

for year_start, (letter, year_label) in CYCLES.items():
    demo_p = OUTPUT_DIR / f"DEMO_{letter}_{year_label.replace('-','_')}.XPT"
    mcq_p  = OUTPUT_DIR / f"MCQ_{letter}_{year_label.replace('-','_')}.XPT"
    cbc_p  = OUTPUT_DIR / f"CBC_{letter}_{year_label.replace('-','_')}.XPT"

    if not (demo_p.exists() and mcq_p.exists()):
        print(f"  {year_label}: missing files")
        continue
    try:
        demo = pd.read_sas(demo_p, encoding="utf-8")
        mcq  = pd.read_sas(mcq_p,  encoding="utf-8")
        n    = len(demo)
        nc   = int((mcq["MCQ220"] == 1).sum()) if "MCQ220" in mcq.columns else 0
        grand_n += n
        grand_cancer += nc

        cbc_cols = []
        if cbc_p.exists():
            cbc      = pd.read_sas(cbc_p, encoding="utf-8")
            cbc_cols = [c for c in cbc.columns if c not in ("SEQN", "WTSB2YR")][:8]

        print(f"\n  {year_label}:")
        print(f"    Participants     : {n:,}")
        print(f"    Cancer cases     : {nc:,}  ({100*nc/n:.1f}%)")
        print(f"    CBC columns      : {cbc_cols}")
        print(f"    MCQ columns      : {[c for c in mcq.columns if 'MCQ22' in c or 'MCQ23' in c]}")
    except Exception as e:
        print(f"  {year_label}: read error - {e}")

print(f"\n{'='*65}")
print(f"GRAND TOTAL ACROSS ALL CYCLES:")
print(f"  Total participants  : {grand_n:,}")
print(f"  Total cancer cases  : {grand_cancer:,}")
if grand_n:
    print(f"  Cancer rate         : {100*grand_cancer/grand_n:.1f}%")
print(f"\nSaved to: {OUTPUT_DIR.resolve()}")
