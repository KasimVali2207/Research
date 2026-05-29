"""
Finalize results from however many patients are done in agent_results_100.json
Updates README, full_results_summary.json, and regenerates figures.
"""
import json, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")

with open("results/agent_results_100.json") as f:
    recs = json.load(f)

n = len(recs)
eas_j  = [r["eas_jaccard"]        for r in recs]
eas_o  = [r["eas_overlap_k"]      for r in recs]
hall   = [r["hallucination_rate"] for r in recs]
triags = [r["triage"]             for r in recs]
ctypes = [r["cancer_type"]        for r in recs]
modes  = Counter(r.get("mode","5agent") for r in recs)

mean_eas_j = round(np.mean(eas_j), 4)
std_eas_j  = round(np.std(eas_j),  4)
mean_eas_o = round(np.mean(eas_o), 4)
std_eas_o  = round(np.std(eas_o),  4)
mean_hall  = round(np.mean(hall),  4)
std_hall   = round(np.std(hall),   4)
triage_dist= dict(Counter(triags))
ct_counts  = dict(Counter(ctypes))

print(f"=== FINAL RESULTS (n={n} patients) ===")
print(f"EAS Jaccard:   {mean_eas_j} +/- {std_eas_j}")
print(f"EAS Overlap@5: {mean_eas_o} +/- {std_eas_o}")
print(f"Hallucination: {mean_hall} +/- {std_hall}")
print(f"Triage: {triage_dist}")
print(f"Types:  {ct_counts}")
print(f"Modes:  {dict(modes)}")

# per cancer type
ct_order = ["lung","liver","colorectal","none"]
ct_eas   = {}
ct_hall  = {}
for ct in ct_order:
    subset = [r for r in recs if r["cancer_type"]==ct]
    if subset:
        ct_eas[ct]  = round(np.mean([r["eas_jaccard"]        for r in subset]),4)
        ct_hall[ct] = round(np.mean([r["hallucination_rate"] for r in subset]),4)
        print(f"  {ct:<12}: n={len(subset)}  EAS-J={ct_eas[ct]}  Hall={ct_hall[ct]}")

# ── Update full_results_summary.json ─────────────────────────────────────────
summary = json.load(open("results/full_results_summary.json"))
summary["agentic"]["n_patients_run"]         = n
summary["agentic"]["mean_eas_jaccard"]        = mean_eas_j
summary["agentic"]["mean_eas_jaccard_sd"]     = std_eas_j
summary["agentic"]["mean_eas_overlap_k"]      = mean_eas_o
summary["agentic"]["mean_eas_overlap_k_sd"]   = std_eas_o
summary["agentic"]["mean_hallucination_rate"] = mean_hall
summary["agentic"]["mean_hallucination_sd"]   = std_hall
summary["agentic"]["triage_distribution"]     = triage_dist
summary["agentic"]["eas_by_cancer_type"]      = ct_eas
summary["agentic"]["hallucination_by_cancer_type"] = ct_hall
summary["agentic"]["mode_breakdown"]          = dict(modes)
summary["agentic"]["note"] = (
    f"Evaluation on n={n} real NHANES patients "
    f"(cancer: lung={ct_counts.get('lung',0)}, liver={ct_counts.get('liver',0)}, "
    f"colorectal={ct_counts.get('colorectal',0)}; control={ct_counts.get('none',0)}). "
    f"Mode: {dict(modes)}. "
    "Formal hallucination scorer (regex ±15% tolerance). "
    "LLaMA 3.3 70B via Groq.")
with open("results/full_results_summary.json","w") as f:
    json.dump(summary, f, indent=2)
print(f"\nUpdated: full_results_summary.json (n={n})")

# ── Regenerate figures ─────────────────────────────────────────────────────────
FIG = Path("results/figures")
plt.rcParams.update({"figure.dpi":150,"font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
COLORS = {"lung":"#2196F3","liver":"#FF5722","colorectal":"#4CAF50","none":"#9E9E9E"}

fig, axes = plt.subplots(1,3,figsize=(15,5))
axes[0].hist(eas_j, bins=15, color="#2196F3", edgecolor="white", alpha=0.85)
axes[0].axvline(mean_eas_j, color="#FF5722", lw=2, label=f"Mean={mean_eas_j}")
axes[0].axvline(0.05, color="gray",  lw=1, ls="--", label="Poor (<0.05)")
axes[0].axvline(0.15, color="green", lw=1, ls="--", label="Good (>0.15)")
axes[0].set_xlabel("EAS Jaccard"); axes[0].set_ylabel("Count")
axes[0].set_title(f"EAS Jaccard Distribution\n(n={n}, mean={mean_eas_j}±{std_eas_j})")
axes[0].legend(fontsize=8)

axes[1].hist(eas_o, bins=15, color="#4CAF50", edgecolor="white", alpha=0.85)
axes[1].axvline(mean_eas_o, color="#FF5722", lw=2, label=f"Mean={mean_eas_o}")
axes[1].set_xlabel("EAS Overlap@5"); axes[1].set_ylabel("Count")
axes[1].set_title(f"EAS Overlap@5 Distribution\n(n={n}, mean={mean_eas_o}±{std_eas_o})")
axes[1].legend(fontsize=8)

axes[2].hist(hall, bins=15, color="#FF9800", edgecolor="white", alpha=0.85)
axes[2].axvline(mean_hall, color="#FF5722", lw=2, label=f"Mean={mean_hall}")
axes[2].set_xlabel("Hallucination Rate"); axes[2].set_ylabel("Count")
axes[2].set_title(f"Hallucination Distribution\n(n={n}, mean={mean_hall}±{std_hall})")
axes[2].legend(fontsize=8)

plt.suptitle(f"LLM Agent Pipeline Evaluation — n={n} Real NHANES Patients\n"
             f"LLaMA 3.3 70B | EAS = SHAP-LLM Feature Alignment | Formal Hallucination Scorer",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG/"fig29_eas_distribution_n100.png", bbox_inches="tight")
plt.close()
print("Saved: fig29_eas_distribution_n100.png")

# Cancer type boxplot
valid_ct = {ct:{"eas":[r["eas_jaccard"] for r in recs if r["cancer_type"]==ct],
                "hall":[r["hallucination_rate"] for r in recs if r["cancer_type"]==ct]}
            for ct in ct_order if any(r["cancer_type"]==ct for r in recs)}

fig, axes = plt.subplots(1,2,figsize=(12,5))
labels = list(valid_ct.keys())
bp = axes[0].boxplot([valid_ct[ct]["eas"] for ct in labels], labels=labels, patch_artist=True)
for p,ct in zip(bp["boxes"],labels): p.set_facecolor(COLORS.get(ct,"#9E9E9E")); p.set_alpha(0.7)
axes[0].axhline(mean_eas_j, ls="--", color="red", lw=1.5, label=f"Overall={mean_eas_j}")
axes[0].set_ylabel("EAS Jaccard"); axes[0].set_title(f"EAS by Cancer Type (n={n})")
axes[0].legend(fontsize=8)
for i,(ct,d) in enumerate(valid_ct.items()):
    axes[0].text(i+1, max(d["eas"])+0.01, f"n={len(d['eas'])}", ha="center", fontsize=8)

bp2= axes[1].boxplot([valid_ct[ct]["hall"] for ct in labels], labels=labels, patch_artist=True)
for p,ct in zip(bp2["boxes"],labels): p.set_facecolor(COLORS.get(ct,"#9E9E9E")); p.set_alpha(0.7)
axes[1].axhline(mean_hall, ls="--", color="red", lw=1.5, label=f"Overall={mean_hall}")
axes[1].set_ylabel("Hallucination Rate"); axes[1].set_title(f"Hallucination by Cancer Type (n={n})")
axes[1].legend(fontsize=8)

plt.suptitle(f"Agent Performance by Cancer Type — n={n} Real Patients", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG/"fig31_eas_by_cancer_type_n100.png", bbox_inches="tight")
plt.close()
print("Saved: fig31_eas_by_cancer_type_n100.png")

# Print values for README update
print(f"\n{'='*55}")
print(f"VALUES FOR README ABLATION TABLE (n={n} real patients):")
print(f"  Full Pipeline  EAS-J={mean_eas_j}  EAS-O5={mean_eas_o}  Hall={mean_hall}")
print(f"  Cancer: {ct_counts}")
print(f"  95% CI EAS-J (bootstrap): [{round(mean_eas_j-1.96*std_eas_j/np.sqrt(n),4)}, {round(mean_eas_j+1.96*std_eas_j/np.sqrt(n),4)}]")
