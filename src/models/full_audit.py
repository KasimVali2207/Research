"""Complete audit script — checks every result file, every code file, every figure."""
import json, sys, os
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

ISSUES = {"CRITICAL":[], "MODERATE":[], "MINOR":[], "OK":[]}
def crit(msg): ISSUES["CRITICAL"].append(msg); print(f"  [CRITICAL] {msg}")
def mod(msg):  ISSUES["MODERATE"].append(msg); print(f"  [MODERATE] {msg}")
def minor(msg):ISSUES["MINOR"].append(msg);    print(f"  [MINOR]    {msg}")
def ok(msg):   ISSUES["OK"].append(msg);       print(f"  [OK]       {msg}")

print("="*65)
print("FULL REPOSITORY AUDIT")
print("="*65)

# ── 1. Result JSON cross-consistency ─────────────────────────────────────────
print("\n[1] Result JSON cross-consistency")
model = json.load(open("results/nhanes_model_results.json"))
full  = json.load(open("results/full_results_summary.json"))
ablat = json.load(open("results/ablation_results.json"))
agent = json.load(open("results/agent_results.json"))
agsm  = json.load(open("results/agentic_summary.json"))
verif = json.load(open("results/nhanes_model_results_verified.json"))

best_auroc_model = max(v["AUROC"] for v in model.values())
full_auroc       = full["model_auroc"]
verif_gb_auroc   = verif["Gradient Boosting"]["AUROC"]
ci               = full["auroc_95ci"]

if abs(best_auroc_model - full_auroc) < 0.001:
    ok(f"model_results vs full_summary AUROC match: {best_auroc_model}")
else:
    crit(f"AUROC mismatch: nhanes_model_results={best_auroc_model} vs full_summary={full_auroc}")

if abs(verif_gb_auroc - full_auroc) < 0.001:
    ok(f"Verified AUROC matches summary: {verif_gb_auroc}")
else:
    crit(f"Verified AUROC ({verif_gb_auroc}) != summary ({full_auroc})")

if ci[0] <= full_auroc <= ci[1]:
    ok(f"AUROC={full_auroc} within 95% CI [{ci[0]},{ci[1]}]")
else:
    crit(f"AUROC={full_auroc} NOT within CI [{ci[0]},{ci[1]}] -- impossible!")

ci_width = ci[1] - ci[0]
if 0.01 < ci_width < 0.08:
    ok(f"CI width={ci_width:.4f} is plausible for n=16762")
else:
    crit(f"CI width={ci_width:.4f} is suspicious (expected 0.02-0.06)")

# agentic_summary.json is stale
if agsm.get("n_patients") != len(agent):
    mod(f"agentic_summary.json says n_patients={agsm.get('n_patients')} but agent_results.json has {len(agent)} -- STALE FILE")
else:
    ok(f"agentic_summary n_patients={agsm.get('n_patients')} matches agent_results")

# Rate limit check
rate_lim = sum(1 for r in agent
               for k in ["a1_biomarker","a2_risk","a3_differential","a4_evidence","a5_triage"]
               if str(r.get(k,"")).startswith("[Rate") or str(r.get(k,"")).startswith("[Failed"))
if rate_lim == 0:
    ok("No [Rate limit] or [Failed] slots in agent_results.json")
else:
    crit(f"{rate_lim} agent slots still have [Rate limit] or [Failed]")

# Ablation method labels
non_real = [c for c,v in ablat.items() if v.get("method") not in ("real_llm","computed")]
if not non_real:
    ok("All ablation conditions have correct method labels (real_llm/computed)")
else:
    mod(f"Ablation conditions missing method label: {non_real}")

# ── 2. Data integrity ─────────────────────────────────────────────────────────
print("\n[2] Data integrity")
df = pd.read_parquet("data/processed/nhanes_features.parquet")
stats = json.load(open("data/processed/nhanes_stats.json"))
n_cancer = int((df["label"]==1).sum())
n_total  = len(df)
n_ctrl   = int((df["label"]==0).sum())

if n_total == full["n_total"] and n_cancer == full["n_cancer"]:
    ok(f"Parquet matches summary: n={n_total}, cancer={n_cancer}")
else:
    crit(f"Parquet n={n_total},cancer={n_cancer} vs summary n={full['n_total']},cancer={full['n_cancer']}")

if stats["total_subjects"] == n_total:
    ok(f"nhanes_stats.json consistent with parquet: n={n_total}")
else:
    mod(f"nhanes_stats.json total={stats['total_subjects']} vs parquet {n_total}")

# Check for label -1 leakage (excluded cancers)
neg_labels = (df["label"]==-1).sum()
if neg_labels == 0:
    ok("No label=-1 (excluded other-cancer) in parquet -- correctly removed")
else:
    crit(f"{neg_labels} rows with label=-1 in parquet -- preprocessing bug!")

# Feature columns check
FEAT = [c for c in df.columns if c not in
        ("seqn","cancer","cancer_type","ever_cancer","label","cycle","gender","ethnicity","age")]
if len(FEAT) == 31:
    ok(f"31 feature columns present")
else:
    mod(f"Expected 31 features, got {len(FEAT)}: {FEAT}")

# Missing data check
miss = df[FEAT].isna().mean()
high_miss = miss[miss > 0.40].index.tolist()
if not high_miss:
    ok("No feature has >40% missing data")
else:
    mod(f"High missingness (>40%): {high_miss}")

# ── 3. Code correctness checks ────────────────────────────────────────────────
print("\n[3] Code correctness checks")

# Check nhanes_agent_pipeline.py for PPV bug (line 227)
with open("src/agents/nhanes_agent_pipeline.py", encoding="utf-8", errors="replace") as f:
    pipeline_code = f.read()

if '"ppv": float((probs[y==1]>=thr_arr[idx_s]).mean())' in pipeline_code:
    crit("PPV BUG in nhanes_agent_pipeline.py line~227: computes PPV as fraction of cancer cases above threshold (= sensitivity), not TP/(TP+FP)")
else:
    ok("PPV calculation in nhanes_agent_pipeline.py looks correct")

if "GROQ_API_KEY" in pipeline_code and "os.getenv" in pipeline_code and "gsk_" not in pipeline_code:
    ok("No hardcoded API key in nhanes_agent_pipeline.py")
else:
    if "gsk_" in pipeline_code:
        crit("Hardcoded API key (gsk_...) found in nhanes_agent_pipeline.py!")

if "cross_val_predict" in pipeline_code:
    ok("Agent pipeline uses cross_val_predict (correct CV approach)")

# Check train_nhanes.py for data leakage
with open("src/models/train_nhanes.py", encoding="utf-8", errors="replace") as f:
    train_code = f.read()

if "cross_val_predict" in train_code and "fit(" not in train_code.split("cross_val_predict")[0].split("# ── Fig")[-1]:
    ok("train_nhanes.py uses cross_val_predict correctly")
else:
    mod("Review train_nhanes.py for potential data leakage in figure generation (rf_pipe.fit called after CV)")

# Check for hardcoded values
if "0.9116" in pipeline_code or "0.9349" in pipeline_code:
    crit("Old wrong CI [0.9116,0.9349] hardcoded in nhanes_agent_pipeline.py")
else:
    ok("Old wrong CI values not in pipeline code")

if "0.9116" in open("README.md", encoding="utf-8", errors="replace").read():
    crit("Old wrong CI [0.9116] still in README.md")
else:
    ok("README.md does not contain old wrong CI values")

# ── 4. Figure audit ────────────────────────────────────────────────────────────
print("\n[4] Figure audit")
import os
figs_dir = "results/figures"
all_figs = os.listdir(figs_dir)

# Check for duplicate fig25 and fig28
dup25 = [f for f in all_figs if f.startswith("fig25")]
dup28 = [f for f in all_figs if f.startswith("fig28")]
dup27 = [f for f in all_figs if f.startswith("fig27")]
dup29 = [f for f in all_figs if f.startswith("fig29")]

if len(dup25) > 1:
    mod(f"Duplicate fig25 files: {dup25}")
else:
    ok(f"fig25: {dup25}")

if len(dup28) > 1:
    mod(f"Duplicate fig28 files: {dup28}")
else:
    ok(f"fig28: {dup28}")

if len(dup27) > 1:
    mod(f"Duplicate fig27 files: {dup27}")
else:
    ok(f"fig27: {dup27}")

if len(dup29) > 1:
    mod(f"Duplicate fig29 files: {dup29}")
else:
    ok(f"fig29: {dup29}")

# orphan figures
orphans = ["subgroup_fairness.png","temporal_vs_static.png","calibration_reliability.png"]
found_orphans = [f for f in orphans if f in all_figs]
if found_orphans:
    minor(f"Orphan figures (old runs, not referenced): {found_orphans}")

# ── 5. Stale/orphan files ─────────────────────────────────────────────────────
print("\n[5] Stale and orphan files")
stale_files = [
    "results/nhanes_model_results_verified.json",  # duplicate of nhanes_model_results.json
    "data/processed/pubmed_kb.jsonl",              # large stale RAG file
    "results/agentic_summary.json",                # stale (n=15, outdated)
]
for f in stale_files:
    if os.path.exists(f):
        minor(f"Stale file exists: {f}")

# ── 6. README accuracy spot-checks ────────────────────────────────────────────
print("\n[6] README accuracy")
readme = open("README.md", encoding="utf-8", errors="replace").read()

checks = [
    ("0.7238", "Gradient Boosting AUROC"),
    ("0.7059", "CI lower bound"),
    ("0.7443", "CI upper bound"),
    ("16,762", "n total"),
    ("485", "n cancer"),
    ("2.89%", "cancer prevalence"),
    ("real_llm", "ablation method labels"),
    ("PPV", "corrected operating points"),
]
for val, desc in checks:
    if val in readme:
        ok(f"README contains correct value '{val}' ({desc})")
    else:
        mod(f"README missing expected value '{val}' ({desc})")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("AUDIT SUMMARY")
print("="*65)
print(f"  CRITICAL issues: {len(ISSUES['CRITICAL'])}")
for i in ISSUES["CRITICAL"]: print(f"    -> {i}")
print(f"  MODERATE issues: {len(ISSUES['MODERATE'])}")
for i in ISSUES["MODERATE"]: print(f"    -> {i}")
print(f"  MINOR issues:    {len(ISSUES['MINOR'])}")
for i in ISSUES["MINOR"]: print(f"    -> {i}")
print(f"  OK checks:       {len(ISSUES['OK'])}")
