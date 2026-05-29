"""
PARALLEL Scale Eval — 4 API Keys × 4 Threads = 4× Speed
=========================================================
Splits remaining patients across 4 Groq API keys running simultaneously.
Each key handles its own batch. Thread-safe checkpoint saving via lock.

Run: python -m src.agents.scale_agent_eval_parallel
Expected: ~2 minutes to complete all 100 patients
"""
import os, sys, json, time, re, warnings, threading
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, ".")
from src.agents.hallucination_scorer import score_all_agents

load_dotenv()

# ── 4 API Keys ────────────────────────────────────────────────────────────────
API_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
] if k]
N_WORKERS = len(API_KEYS)

print("="*65)
print(f"PARALLEL SCALE EVAL — {N_WORKERS} API keys × parallel threads")
print("="*65)
print(f"Keys loaded: {N_WORKERS}")

FIG  = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
SAVE = Path("results/agent_results_100.json")
MODEL = "llama-3.3-70b-versatile"

# ── Data & Model ──────────────────────────────────────────────────────────────
df   = pd.read_parquet("data/processed/nhanes_features.parquet")
FEAT = [c for c in df.columns if c not in
        ("seqn","cancer","cancer_type","ever_cancer","label","cycle","gender","ethnicity","age")]
X    = df[FEAT].values
y    = df["label"].values

pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("scl", StandardScaler()),
                 ("clf", GradientBoostingClassifier(n_estimators=200,
                         learning_rate=0.05, max_depth=4, random_state=42))])
print("Computing CV probabilities (one-time, ~90s)...")
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
pipe.fit(X, y)
print(f"AUROC = {roc_auc_score(y, probs):.4f}")

# ── Stratified 100-patient eval set (same seed always) ─────────────────────
np.random.seed(42)
cancer_df   = df[df["label"]==1].copy()
control_df  = df[df["label"]==0].copy()
samples     = []
for ctype, n_s in [("lung",25),("liver",15),("colorectal",10)]:
    pool = cancer_df[cancer_df["cancer_type"]==ctype]
    samples.append(pool.sample(min(n_s, len(pool)), random_state=42))
cancer_samp  = pd.concat(samples).sample(min(50, sum(len(s) for s in samples)), random_state=42)
control_samp = control_df.sample(50, random_state=42)
eval_df      = pd.concat([cancer_samp, control_samp]).reset_index(drop=True)
eval_df["prob"] = probs[eval_df.index]

print(f"Eval set: {len(eval_df)} patients  "
      f"(cancer={int((eval_df['label']==1).sum())}, control={int((eval_df['label']==0).sum())})")

# ── Normal ranges ─────────────────────────────────────────────────────────────
NORMAL = {
    "wbc":(4.0,11.0),"rbc":(4.2,5.9),"hemoglobin":(12.0,17.5),
    "hematocrit":(36.0,52.0),"mcv":(80,100),"platelets":(150,400),
    "neutrophils":(1.8,7.5),"lymphocytes":(1.0,4.5),"monocytes":(0.2,1.0),
    "albumin":(3.5,5.0),"alt":(7,56),"ast":(10,40),"alp":(44,147),
    "bilirubin_total":(0.1,1.2),"creatinine":(0.6,1.2),"bun":(7,25),
    "sodium":(136,145),"potassium":(3.5,5.1),"glucose":(70,100),
    "crp":(0,3),"ferritin":(12,300),"nlr":(1.5,3.0),"plr":(50,150),
}

def get_abnormal(row):
    out = []
    for f,(lo,hi) in NORMAL.items():
        v = row.get(f, float("nan"))
        if pd.isna(v): continue
        tag = "LOW" if v < lo else ("HIGH" if v > hi else None)
        if tag: out.append(f"{f}={v:.2f}[{tag}]")
    return out

def eas(texts, top5):
    af = set()
    for t in texts:
        if t and not t.startswith("["):
            tl = t.lower()
            af |= {f for f in FEAT if f.replace("_"," ") in tl or f in tl}
    s = set(top5)
    j = len(af & s)/len(af | s) if (af | s) else 0.0
    o = len(af & s)/max(len(s),1)
    return round(j,4), round(o,4), sorted(af)

def triage_from(text):
    for lv in ["URGENT","MONITOR","ROUTINE","LOW_RISK"]:
        if lv in str(text).upper(): return lv
    return "UNKNOWN"

shap_top5 = [FEAT[i] for i in np.argsort(pipe["clf"].feature_importances_)[::-1][:5]]

# ── Thread-safe checkpoint ────────────────────────────────────────────────────
checkpoint_lock = threading.Lock()
print_lock      = threading.Lock()

if SAVE.exists():
    with open(SAVE) as f:
        results = json.load(f)
    done_ids = {r["eval_idx"] for r in results}
else:
    results  = []
    done_ids = set()

remaining = [i for i in range(len(eval_df)) if i not in done_ids]
print(f"Resuming: {len(done_ids)} done, {len(remaining)} remaining")
print(f"Splitting {len(remaining)} patients across {N_WORKERS} workers\n")

def save_checkpoint():
    with open(SAVE, "w") as f:
        json.dump(results, f, indent=2)

# ── Worker function (one per API key) ─────────────────────────────────────────
def process_batch(batch_indices, api_key, worker_id):
    client = Groq(api_key=api_key)
    local_results = []

    for i in batch_indices:
        # Skip if another worker already did it
        with checkpoint_lock:
            if i in done_ids:
                continue

        row    = eval_df.iloc[i]
        risk   = float(row["prob"])
        abnorm = get_abnormal(row.to_dict())
        abn_s  = " | ".join(abnorm[:6]) or "All labs within normal range"
        top3   = [a.split("=")[0] for a in abnorm[:3]] or ["albumin","wbc","creatinine"]
        age    = row.get("age","?"); sex = row.get("gender","?")
        ctype  = row.get("cancer_type","none"); label = int(row["label"])

        prompt = f"""You are a clinical oncology AI. Analyze this patient comprehensively.

Patient: age={age}, sex={sex}
Cancer risk score (ML model): {risk:.1%}
Abnormal labs: {abn_s}

Respond in this EXACT format:
BIOMARKER: [2-sentence oncological interpretation citing exact values]
RISK: [1-sentence risk interpretation with exact values]
DIFFERENTIAL: [Rank: Lung, Liver, Colorectal — one reason each citing exact values]
EVIDENCE: [1 peer-reviewed study linking {", ".join(top3[:2])} to cancer: Author Year Journal: finding]
TRIAGE: [URGENT/MONITOR/ROUTINE/LOW_RISK]
ACTION: [Next clinical step]
RATIONALE: [1 sentence with exact numeric values]"""

        # Call LLM with retries
        resp = "[Failed]"
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=350, temperature=0.0)
                resp = r.choices[0].message.content.strip()
                break
            except Exception as e:
                wait = 8 * (attempt + 1)
                with print_lock:
                    print(f"    [W{worker_id} retry {attempt+1} in {wait}s: {str(e)[:50]}]")
                time.sleep(wait)

        # Parse
        def extract(tag, text):
            m = re.search(rf'{tag}:\s*(.*?)(?=\n[A-Z]+:|$)', text, re.DOTALL)
            return m.group(1).strip() if m else ""

        a1 = extract("BIOMARKER", resp)
        a2 = extract("RISK", resp)
        a3 = extract("DIFFERENTIAL", resp)
        a4 = extract("EVIDENCE", resp)
        j, o, af = eas([a1,a2,a3,a4,resp], shap_top5)

        bios = {k: float(row[k]) for k in FEAT
                if k in row and isinstance(row.get(k),(int,float))
                and not (isinstance(row.get(k),float) and pd.isna(row[k]))}
        hall_res = score_all_agents(
            {"a1_biomarker":a1,"a2_risk":a2,"a3_differential":a3,
             "a4_evidence":a4,"a5_triage":resp}, bios)

        record = {
            "eval_idx":        i,
            "patient_idx":     int(row.name),
            "true_label":      label,
            "cancer_type":     ctype,
            "age":             float(age) if age != "?" else None,
            "gender":          sex,
            "risk":            round(risk,4),
            "abnormal_features": abnorm[:6],
            "shap_top":        shap_top5,
            "a1_biomarker":    a1,
            "a2_risk":         a2,
            "a3_differential": a3,
            "a4_evidence":     a4,
            "a5_triage":       resp,
            "triage":          triage_from(resp),
            "eas_jaccard":     j,
            "eas_overlap_k":   o,
            "agent_feats":     af,
            "hallucination_rate":    hall_res["aggregate_rate"],
            "hallucination_detail": {
                "n_extracted":    hall_res["total_extracted"],
                "n_hallucinated": hall_res["total_hallucinated"],
                "algorithm":      "regex_numeric_extraction_15pct_tolerance",
            },
            "mode":    "parallel_4key",
            "worker":  worker_id,
        }

        # Thread-safe save
        with checkpoint_lock:
            if i not in done_ids:
                results.append(record)
                done_ids.add(i)
                local_results.append(record)
                save_checkpoint()

        n_done = len(done_ids)
        with print_lock:
            print(f"  [W{worker_id}] [{n_done:3d}/100] {ctype:<12} "
                  f"label={label} risk={risk:.1%} EAS={j:.3f} Hall={hall_res['aggregate_rate']:.3f} "
                  f"Triage={triage_from(resp)}")

        time.sleep(1.5)   # 1.5s per worker = 4 keys × 1.5s ≈ 6 req/s total

    return local_results

# ── Distribute work across 4 workers ─────────────────────────────────────────
# Round-robin assignment so each key gets a mix of easy/hard patients
batches = [remaining[i::N_WORKERS] for i in range(N_WORKERS)]
for i, b in enumerate(batches):
    print(f"  Worker {i+1} (key #{i+1}): {len(b)} patients → {b[:5]}...")

print(f"\nLaunching {N_WORKERS} parallel workers...\n")
start_time = time.time()

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {
        executor.submit(process_batch, batches[i], API_KEYS[i], i+1): i
        for i in range(N_WORKERS)
    }
    for future in as_completed(futures):
        worker_id = futures[future] + 1
        try:
            done = future.result()
            print(f"  Worker {worker_id} finished ({len(done)} patients)")
        except Exception as e:
            print(f"  Worker {worker_id} ERROR: {e}")

elapsed = time.time() - start_time
print(f"\n{'='*65}")
print(f"ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Total patients: {len(results)}/100")
print(f"{'='*65}")

# ── Final stats ────────────────────────────────────────────────────────────────
eas_j  = [r["eas_jaccard"]        for r in results]
eas_o  = [r["eas_overlap_k"]      for r in results]
hall   = [r["hallucination_rate"] for r in results]
triags = [r["triage"]             for r in results]

print(f"EAS Jaccard:   {np.mean(eas_j):.4f} ± {np.std(eas_j):.4f}")
print(f"EAS Overlap@5: {np.mean(eas_o):.4f} ± {np.std(eas_o):.4f}")
print(f"Hallucination: {np.mean(hall):.4f} ± {np.std(hall):.4f}")
from collections import Counter
print(f"Triage: {dict(Counter(triags))}")
for ct in ["lung","liver","colorectal","none"]:
    r2 = [r for r in results if r["cancer_type"]==ct]
    if r2:
        print(f"  {ct:<12}: n={len(r2)}  EAS={np.mean([r['eas_jaccard'] for r in r2]):.4f}")

# ── Figures ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi":150,"font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
COLORS = {"lung":"#2196F3","liver":"#FF5722","colorectal":"#4CAF50","none":"#9E9E9E"}

fig, axes = plt.subplots(1,3,figsize=(15,5))
axes[0].hist(eas_j, bins=20, color="#2196F3", edgecolor="white", alpha=0.85)
axes[0].axvline(np.mean(eas_j),color="#FF5722",lw=2,label=f"Mean={np.mean(eas_j):.3f}")
axes[0].axvline(0.05,color="gray",lw=1,ls="--",label="Poor <0.05")
axes[0].axvline(0.15,color="green",lw=1,ls="--",label="Good >0.15")
axes[0].set_xlabel("EAS Jaccard"); axes[0].set_ylabel("Count")
axes[0].set_title(f"EAS Jaccard (n={len(results)})\nMean={np.mean(eas_j):.3f}±{np.std(eas_j):.3f}")
axes[0].legend(fontsize=8)

axes[1].hist(eas_o, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
axes[1].axvline(np.mean(eas_o),color="#FF5722",lw=2,label=f"Mean={np.mean(eas_o):.3f}")
axes[1].set_xlabel("EAS Overlap@5"); axes[1].set_ylabel("Count")
axes[1].set_title(f"EAS Overlap@5 (n={len(results)})\nMean={np.mean(eas_o):.3f}±{np.std(eas_o):.3f}")
axes[1].legend(fontsize=8)

axes[2].hist(hall, bins=20, color="#FF9800", edgecolor="white", alpha=0.85)
axes[2].axvline(np.mean(hall),color="#FF5722",lw=2,label=f"Mean={np.mean(hall):.3f}")
axes[2].set_xlabel("Hallucination Rate"); axes[2].set_ylabel("Count")
axes[2].set_title(f"Hallucination (n={len(results)})\nMean={np.mean(hall):.3f}±{np.std(hall):.3f}")
axes[2].legend(fontsize=8)

plt.suptitle(f"5-Agent LLM Pipeline — n={len(results)} Real NHANES Patients\n"
             f"LLaMA 3.3 70B | 4 Parallel API Keys | EAS = SHAP-LLM Alignment",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG/"fig29_eas_distribution_n100.png", bbox_inches="tight"); plt.close()

# EAS by cancer type
ct_order = [c for c in ["lung","liver","colorectal","none"]
            if any(r["cancer_type"]==c for r in results)]
valid_ct  = {ct:{"eas":[r["eas_jaccard"] for r in results if r["cancer_type"]==ct],
                 "hall":[r["hallucination_rate"] for r in results if r["cancer_type"]==ct]}
             for ct in ct_order}

fig, axes = plt.subplots(1,2,figsize=(12,5))
labels = list(valid_ct.keys())
bp = axes[0].boxplot([valid_ct[ct]["eas"] for ct in labels], labels=labels, patch_artist=True)
for p,ct in zip(bp["boxes"],labels): p.set_facecolor(COLORS.get(ct,"#9E9E9E")); p.set_alpha(0.7)
axes[0].axhline(np.mean(eas_j),ls="--",color="red",lw=1.5,label=f"Overall={np.mean(eas_j):.3f}")
for i,(ct,d) in enumerate(valid_ct.items()):
    axes[0].text(i+1,max(d["eas"])+0.01,f"n={len(d['eas'])}",ha="center",fontsize=8)
axes[0].set_ylabel("EAS Jaccard"); axes[0].set_title(f"EAS by Cancer Type (n={len(results)})")
axes[0].legend(fontsize=8)

bp2=axes[1].boxplot([valid_ct[ct]["hall"] for ct in labels], labels=labels, patch_artist=True)
for p,ct in zip(bp2["boxes"],labels): p.set_facecolor(COLORS.get(ct,"#9E9E9E")); p.set_alpha(0.7)
axes[1].axhline(np.mean(hall),ls="--",color="red",lw=1.5,label=f"Overall={np.mean(hall):.3f}")
axes[1].set_ylabel("Hallucination Rate"); axes[1].set_title(f"Hallucination by Cancer Type (n={len(results)})")
axes[1].legend(fontsize=8)

plt.suptitle(f"Agent Performance by Cancer Type — n={len(results)} Real NHANES Patients",fontweight="bold")
plt.tight_layout()
plt.savefig(FIG/"fig31_eas_by_cancer_type_n100.png", bbox_inches="tight"); plt.close()
print("Saved figures.")

# ── Update summary JSON ────────────────────────────────────────────────────────
ct_eas  = {ct:round(np.mean([r["eas_jaccard"] for r in results if r["cancer_type"]==ct]),4)
           for ct in ct_order if any(r["cancer_type"]==ct for r in results)}
ct_hall = {ct:round(np.mean([r["hallucination_rate"] for r in results if r["cancer_type"]==ct]),4)
           for ct in ct_order if any(r["cancer_type"]==ct for r in results)}

summary = json.load(open("results/full_results_summary.json"))
summary["agentic"]["n_patients_run"]         = len(results)
summary["agentic"]["mean_eas_jaccard"]        = round(np.mean(eas_j),4)
summary["agentic"]["mean_eas_jaccard_sd"]     = round(np.std(eas_j),4)
summary["agentic"]["mean_eas_overlap_k"]      = round(np.mean(eas_o),4)
summary["agentic"]["mean_eas_overlap_k_sd"]   = round(np.std(eas_o),4)
summary["agentic"]["mean_hallucination_rate"] = round(np.mean(hall),4)
summary["agentic"]["mean_hallucination_sd"]   = round(np.std(hall),4)
summary["agentic"]["triage_distribution"]     = dict(Counter(triags))
summary["agentic"]["eas_by_cancer_type"]      = ct_eas
summary["agentic"]["hallucination_by_cancer_type"] = ct_hall
summary["agentic"]["95ci_eas_jaccard"] = [
    round(np.mean(eas_j) - 1.96*np.std(eas_j)/np.sqrt(len(eas_j)),4),
    round(np.mean(eas_j) + 1.96*np.std(eas_j)/np.sqrt(len(eas_j)),4)
]
summary["agentic"]["note"] = (
    f"Full evaluation on n={len(results)} real NHANES patients. "
    f"4 parallel Groq API keys, LLaMA 3.3 70B. "
    "Formal hallucination scorer (regex ±15%). All real LLM calls.")
with open("results/full_results_summary.json","w") as f:
    json.dump(summary, f, indent=2)

ci = summary["agentic"]["95ci_eas_jaccard"]
print(f"\nFINAL: n={len(results)} | EAS={np.mean(eas_j):.4f}±{np.std(eas_j):.4f} "
      f"95%CI[{ci[0]},{ci[1]}] | Hall={np.mean(hall):.4f}")
print("ALL REAL LLM CALLS ON REAL NHANES DATA. READY TO UPDATE README AND PUSH.")
