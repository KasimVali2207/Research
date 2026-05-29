"""
FINAL AUDIT — Research_biomedical repo
Checks every critical component for publication readiness.
"""
import json, sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

PASS = "PASS"; FAIL = "FAIL"; WARN = "WARN"
issues = []

def check(label, condition, msg="", level=FAIL):
    status = PASS if condition else level
    icon = "✓" if condition else ("⚠" if level==WARN else "✗")
    print(f"  {icon} [{status}] {label}" + (f" — {msg}" if msg else ""))
    if not condition:
        issues.append((level, label, msg))

print("="*65)
print("FINAL AUDIT — Research_biomedical")
print("="*65)

# ── 1. RESULT FILES ────────────────────────────────────────────────
print("\n[1] RESULT FILES")
required = {
    "results/nhanes_model_results.json": 100,
    "results/agent_results.json": 10000,
    "results/agent_results_100.json": 50000,
    "results/ablation_results.json": 1000,
    "results/full_results_summary.json": 1000,
    "results/explainability_comparison.json": 500,
}
for path, min_bytes in required.items():
    p = Path(path)
    check(path, p.exists() and p.stat().st_size >= min_bytes,
          f"size={p.stat().st_size if p.exists() else 0}b (min={min_bytes}b)")

# ── 2. FIGURES ────────────────────────────────────────────────────
print("\n[2] FIGURES")
figs = sorted(Path("results/figures").glob("*.png"))
check("Total figures >= 38", len(figs) >= 38, f"found {len(figs)}")
critical_figs = [
    "fig01_roc_curves.png","fig06_calibration.png",
    "fig25_bootstrap_ci.png","fig26_decision_curve_analysis.png",
    "fig27_permutation_test.png","fig28_roc_clinical_operating_points.png",
    "fig29_eas_distribution_n100.png","fig31_eas_by_cancer_type_n100.png",
    "fig37_ablation_study.png","fig38_shap_vs_lime.png",
]
for f in critical_figs:
    p = Path("results/figures") / f
    check(f, p.exists() and p.stat().st_size > 10000, f"size={p.stat().st_size if p.exists() else 0}b")

# ── 3. KEY NUMBERS ─────────────────────────────────────────────────
print("\n[3] KEY NUMBERS")
with open("results/full_results_summary.json") as f:
    s = json.load(f)

auroc = s.get("model_auroc")
ci    = s.get("auroc_95ci", [])
pval  = s.get("permutation_pval")
ag    = s.get("agentic", {})
cop   = s.get("clinical_operating_points", {})

check("AUROC = 0.7238",      auroc == 0.7238,   f"got {auroc}")
check("Bootstrap CI correct", ci == [0.7059, 0.7443], f"got {ci}")
check("Permutation p < 0.001", pval is not None and pval < 0.001, f"got {pval}")
check("LLM n >= 50",          ag.get("n_patients_run",0) >= 50,   f"got n={ag.get('n_patients_run')}")
check("EAS Jaccard exists",   "mean_eas_jaccard" in ag,            f"val={ag.get('mean_eas_jaccard')}")
check("Hallucination exists", "mean_hallucination_rate" in ag,     f"val={ag.get('mean_hallucination_rate')}")
check("95% CI EAS exists",    "95ci_eas_jaccard" in ag,            f"val={ag.get('95ci_eas_jaccard')}")
check("PPV at 80% spec",      "0.8" in cop and cop["0.8"].get("ppv",1) < 0.15,
      f"PPV={cop.get('0.8',{}).get('ppv','?')}")
check("PPV != sensitivity",   "0.8" in cop and cop["0.8"].get("ppv") != cop["0.8"].get("sensitivity"),
      "PPV bug check")

# ── 4. SCRIPTS ────────────────────────────────────────────────────
print("\n[4] SCRIPTS")
scripts = {
    "download_nhanes.py": 500,
    "src/preprocessing/nhanes_to_features.py": 1000,
    "src/models/train_nhanes.py": 5000,
    "src/models/ablation_study.py": 3000,
    "src/agents/nhanes_agent_pipeline.py": 10000,
    "src/agents/hallucination_scorer.py": 3000,
    "src/agents/scale_agent_eval_parallel.py": 5000,
    "src/explainability/lime_comparison.py": 4000,
    "src/models/finalize_results.py": 2000,
    ".env.example": 30,
}
for path, min_bytes in scripts.items():
    p = Path(path)
    check(path, p.exists() and p.stat().st_size >= min_bytes,
          f"size={p.stat().st_size if p.exists() else 0}b")

# ── 5. SECURITY ───────────────────────────────────────────────────
print("\n[5] SECURITY")
key_pat = re.compile(r'gsk_[A-Za-z0-9]{30,}')
hardcoded = [str(p) for p in Path("src").rglob("*.py")
             if key_pat.search(p.read_text(errors="ignore"))]
check("No API keys in .py files", len(hardcoded)==0,
      f"FOUND IN: {hardcoded}" if hardcoded else "")
gitignore = Path(".gitignore").read_text(errors="ignore") if Path(".gitignore").exists() else ""
check(".env in .gitignore",       ".env" in gitignore)
check(".env exists locally",      Path(".env").exists())
check(".env.example exists",      Path(".env.example").exists())

# ── 6. README CHECKS ──────────────────────────────────────────────
print("\n[6] README CONTENT")
readme = Path("README.md").read_text(encoding="utf-8", errors="ignore")
checks_readme = [
    ("Title updated",              "Explanation Alignment Score (EAS)" in readme),
    ("Abstract present",          "Abstract" in readme),
    ("Cross-sectional disclaimer","cross-sectional" in readme.lower()),
    ("EAS definition present",    "EAS_Jaccard" in readme),
    ("Formal hallucination def",  "regex" in readme.lower() and "15%" in readme),
    ("PPV correctly noted",       "TP/(TP+FP)" in readme),
    ("Related Work section",      "Related Work" in readme),
    ("Missing data disclosed",    "CRP" in readme and "42%" in readme),
    ("NNS column in table",       "NNS" in readme),
    ("n=100 in ablation table",   "100" in readme and "Full 5-Agent" in readme),
    ("SHAP vs LIME results",      "τ=0.613" in readme),
    ("LIME comparison figure",    "fig38_shap_vs_lime" in readme),
    ("Limitations section",       "Limitations" in readme),
    ("Citation block",            "bibtex" in readme.lower()),
    ("Apache license",            "Apache" in readme),
]
for label, cond in checks_readme:
    check(label, cond)

# ── 7. GIT SYNC ───────────────────────────────────────────────────
print("\n[7] GIT STATUS")
import subprocess
r = subprocess.run(["git","status","--porcelain"], capture_output=True, text=True)
unstaged = r.stdout.strip()
check("No uncommitted changes", unstaged == "", f"Dirty: {unstaged[:100]}" if unstaged else "")
r2 = subprocess.run(["git","log","--oneline","-1"], capture_output=True, text=True)
check("Latest commit exists", bool(r2.stdout.strip()), r2.stdout.strip())

# ── SUMMARY ───────────────────────────────────────────────────────
print("\n" + "="*65)
fails = [i for i in issues if i[0]==FAIL]
warns = [i for i in issues if i[0]==WARN]
print(f"AUDIT COMPLETE: {len(fails)} FAILURES  |  {len(warns)} WARNINGS")
if fails:
    print("\nFAILURES TO FIX:")
    for _,l,m in fails: print(f"  ✗ {l}: {m}")
if warns:
    print("\nWARNINGS:")
    for _,l,m in warns: print(f"  ⚠ {l}: {m}")
if not fails and not warns:
    print("\n✓ ALL CHECKS PASSED — REPO IS PUBLICATION READY")
elif not fails:
    print("\n✓ NO FAILURES — minor warnings only")
print("="*65)
