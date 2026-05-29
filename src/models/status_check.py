import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
root = Path(".")

# 1. Result files
results_needed = [
    "results/nhanes_model_results.json",
    "results/agent_results.json",
    "results/agent_results_100.json",
    "results/ablation_results.json",
    "results/full_results_summary.json",
    "results/explainability_comparison.json",
]
print("=== RESULT FILES ===")
for r in results_needed:
    p = Path(r)
    status = "OK" if p.exists() else "MISSING"
    size = p.stat().st_size if p.exists() else 0
    print(f"  {status}  {r}  ({size} bytes)")

# 2. Figures
figs = sorted(Path("results/figures").glob("*.png"))
print(f"\n=== FIGURES: {len(figs)} total ===")
for f in figs:
    print(f"  {f.name}")

# 3. Scripts
scripts_needed = [
    "download_nhanes.py",
    "src/preprocessing/nhanes_to_features.py",
    "src/models/train_nhanes.py",
    "src/models/ablation_study.py",
    "src/agents/nhanes_agent_pipeline.py",
    "src/agents/hallucination_scorer.py",
    "src/agents/scale_agent_eval_fast.py",
    "src/explainability/lime_comparison.py",
    "src/models/finalize_results.py",
]
print("\n=== SCRIPTS ===")
for s in scripts_needed:
    p = Path(s)
    status = "OK" if p.exists() else "MISSING"
    print(f"  {status}  {s}")

# 4. Key numbers
print("\n=== KEY NUMBERS ===")
with open("results/full_results_summary.json") as f:
    s = json.load(f)
ml = s.get("ml_performance", {})
ag = s.get("agentic", {})
print(f"  AUROC:         {ml.get('best_auroc','?')}  CI={ml.get('auroc_ci_bootstrap','?')}")
print(f"  Permutation p: {ml.get('permutation_p','?')}")
print(f"  AUPRC:         {ml.get('best_auprc','?')}")
print(f"  LLM patients:  {ag.get('n_patients_run','?')}")
print(f"  EAS Jaccard:   {ag.get('mean_eas_jaccard','?')} +/- {ag.get('mean_eas_jaccard_sd','?')}")
print(f"  Hallucination: {ag.get('mean_hallucination_rate','?')} +/- {ag.get('mean_hallucination_sd','?')}")
print(f"  Triage dist:   {ag.get('triage_distribution','?')}")

# 5. .env check (no hardcoded keys)
print("\n=== SECURITY CHECK ===")
import re
key_pattern = re.compile(r'gsk_[A-Za-z0-9]{40,}')
hardcoded = []
for py in Path("src").rglob("*.py"):
    text = py.read_text(errors="ignore")
    if key_pattern.search(text):
        hardcoded.append(str(py))
if hardcoded:
    print(f"  WARNING: API key hardcoded in {hardcoded}")
else:
    print("  OK  No API keys hardcoded in any .py file")
env = Path(".env")
print(f"  OK  .env exists: {env.exists()}, in .gitignore: {'.env' in Path('.gitignore').read_text(errors='ignore') if Path('.gitignore').exists() else 'no .gitignore'}")
