"""
NHANES ML Training Pipeline
Trains 5 models on real NHANES data, evaluates, generates 20+ publication figures.
"""
import warnings, json, os
warnings.filterwarnings("ignore")
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/models",  exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve,
                              confusion_matrix, classification_report,
                              f1_score, brier_score_loss)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
try:
    import xgboost as xgb; HAS_XGB = True
except: HAS_XGB = False
try:
    import lightgbm as lgb; HAS_LGB = True
except: HAS_LGB = False

# ── Styling ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})
PALETTE = ["#2196F3","#4CAF50","#FF5722","#9C27B0","#FF9800","#00BCD4"]

print("="*65)
print("NHANES CANCER DETECTION — ML Training on REAL Data")
print("="*65)

# ── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/processed/nhanes_features.parquet")
stats = json.load(open("data/processed/nhanes_stats.json"))

FEATURE_COLS = [c for c in df.columns if c not in
    ("seqn","cancer","cancer_type","ever_cancer","label","cycle","gender","ethnicity","age")]
LABEL = "label"

X = df[FEATURE_COLS].values
y = df[LABEL].values
meta = df[["age","gender","ethnicity","cancer_type","cycle"]].copy()

print(f"\nDataset: {len(df):,} subjects | {int(y.sum())} cancer | {int((y==0).sum())} controls")
print(f"Features: {len(FEATURE_COLS)}")
print(f"Cancer types: {df[df.label==1].cancer_type.value_counts().to_dict()}")

# ── Models ───────────────────────────────────────────────────────────────────
imputer  = SimpleImputer(strategy="median")
scaler   = StandardScaler()

def make_pipeline(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler()),
                     ("clf", model)])

models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)),
    "Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                               n_jobs=-1, random_state=42)),
    "Gradient Boosting": make_pipeline(
        GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, random_state=42)),
}
if HAS_XGB:
    sw = (y==0).sum() / (y==1).sum()
    models["XGBoost"] = make_pipeline(
        xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                           scale_pos_weight=sw, eval_metric="aucpr",
                           use_label_encoder=False, random_state=42, verbosity=0))
if HAS_LGB:
    models["LightGBM"] = make_pipeline(
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced", random_state=42, verbose=-1))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Cross-validated predictions ───────────────────────────────────────────────
results = {}
probas   = {}
print("\nTraining models (5-fold CV)...")
for name, pipe in models.items():
    print(f"  {name}...", end=" ", flush=True)
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:,1]
    probas[name] = prob
    auroc = roc_auc_score(y, prob)
    auprc = average_precision_score(y, prob)
    pred  = (prob >= 0.5).astype(int)
    f1    = f1_score(y, pred, zero_division=0)
    brier = brier_score_loss(y, prob)
    results[name] = {"AUROC": auroc, "AUPRC": auprc, "F1": f1, "Brier": brier}
    print(f"AUROC={auroc:.3f}  AUPRC={auprc:.3f}  F1={f1:.3f}")

# Best model
best_name = max(results, key=lambda k: results[k]["AUROC"])
best_prob  = probas[best_name]
print(f"\nBest model: {best_name} (AUROC={results[best_name]['AUROC']:.3f})")

# ── Save results JSON ─────────────────────────────────────────────────────────
with open("results/nhanes_model_results.json","w") as f:
    json.dump({k:{m:round(v,4) for m,v in rv.items()} for k,rv in results.items()}, f, indent=2)
print("Saved: results/nhanes_model_results.json")

# ════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════
FIG = "results/figures"
names = list(models.keys())
colors = PALETTE[:len(names)]

# ── Fig 1: ROC Curves (all models) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,6))
for (name, prob), col in zip(probas.items(), colors):
    fpr, tpr, _ = roc_curve(y, prob)
    auc = results[name]["AUROC"]
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models (Real NHANES Data)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/fig01_roc_curves.png"); plt.close()
print("Fig 01: ROC curves")

# ── Fig 2: PR Curves ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,6))
for (name, prob), col in zip(probas.items(), colors):
    prec, rec, _ = precision_recall_curve(y, prob)
    ap = results[name]["AUPRC"]
    ax.plot(rec, prec, color=col, lw=2, label=f"{name} (AP={ap:.3f})")
baseline = y.mean()
ax.axhline(baseline, color="gray", ls="--", label=f"Baseline ({baseline:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All Models")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/fig02_pr_curves.png"); plt.close()
print("Fig 02: PR curves")

# ── Fig 3: AUROC Bar Chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
vals = [results[n]["AUROC"] for n in names]
bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.6)
for bar, val in zip(bars, vals):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1.05); ax.axvline(0.5, color="red", ls="--", alpha=0.5, label="Random")
ax.set_xlabel("AUROC"); ax.set_title("Model AUROC Comparison (Real NHANES Data)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig03_auroc_bar.png"); plt.close()
print("Fig 03: AUROC bar chart")

# ── Fig 4: AUPRC Bar Chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
vals = [results[n]["AUPRC"] for n in names]
bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.6)
for bar, val in zip(bars, vals):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10)
ax.set_xlim(0, 0.5); ax.set_xlabel("AUPRC")
ax.set_title("Model AUPRC Comparison")
plt.tight_layout(); plt.savefig(f"{FIG}/fig04_auprc_bar.png"); plt.close()
print("Fig 04: AUPRC bar")

# ── Fig 5: Multi-metric comparison ───────────────────────────────────────────
metrics = ["AUROC","AUPRC","F1","Brier"]
fig, axes = plt.subplots(1,4,figsize=(14,5))
for ax, metric in zip(axes, metrics):
    vals = [results[n][metric] for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=8)
    ax.set_title(metric); ax.set_ylim(0, max(vals)*1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", fontsize=8)
plt.suptitle("All Metrics Comparison — Real NHANES Data", fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig(f"{FIG}/fig05_all_metrics.png", bbox_inches="tight"); plt.close()
print("Fig 05: Multi-metric")

# ── Fig 6: Calibration Plot (best model) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,6))
frac_pos, mean_pred = calibration_curve(y, best_prob, n_bins=10)
ax.plot(mean_pred, frac_pos, "s-", color="#2196F3", lw=2, label=f"{best_name}")
ax.plot([0,1],[0,1],"k--", label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title(f"Calibration Plot — {best_name}"); ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/fig06_calibration.png"); plt.close()
print("Fig 06: Calibration")

# ── Fig 7: Confusion Matrix (best model) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,5))
pred = (best_prob >= 0.5).astype(int)
cm   = confusion_matrix(y, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Control","Cancer"], yticklabels=["Control","Cancer"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout(); plt.savefig(f"{FIG}/fig07_confusion_matrix.png"); plt.close()
print("Fig 07: Confusion matrix")

# ── Fig 8: Risk score distribution ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(best_prob[y==0], bins=40, alpha=0.6, color="#2196F3", label="Controls", density=True)
ax.hist(best_prob[y==1], bins=40, alpha=0.6, color="#FF5722", label="Cancer cases", density=True)
ax.set_xlabel("Predicted Cancer Risk Score"); ax.set_ylabel("Density")
ax.set_title(f"Risk Score Distribution — {best_name} (Real NHANES Data)")
ax.legend(); ax.axvline(0.5, color="black", ls="--", label="Threshold=0.5")
plt.tight_layout(); plt.savefig(f"{FIG}/fig08_risk_distribution.png"); plt.close()
print("Fig 08: Risk distribution")

# ── Fig 9: Feature Importance (RF) ───────────────────────────────────────────
rf_pipe = models.get("Random Forest", models[list(models.keys())[0]])
rf_pipe.fit(X, y)
if hasattr(rf_pipe["clf"], "feature_importances_"):
    imp = rf_pipe["clf"].feature_importances_
    feat_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp})
    feat_df = feat_df.sort_values("importance", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8,7))
    bars = ax.barh(feat_df["feature"], feat_df["importance"],
                   color=plt.cm.viridis(np.linspace(0.2,0.9,len(feat_df))))
    ax.set_xlabel("Feature Importance (Gini)"); ax.set_title("Top 20 Feature Importances — Random Forest")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig09_feature_importance.png"); plt.close()
    print("Fig 09: Feature importance")

# ── Fig 10: Cancer type breakdown ────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(12,5))
cancer_df = df[df.label==1]
type_counts = cancer_df["cancer_type"].value_counts()
axes[0].pie(type_counts.values, labels=type_counts.index,
            colors=PALETTE[:len(type_counts)], autopct="%1.1f%%", startangle=90)
axes[0].set_title("Cancer Type Distribution (Real NHANES Data)")
axes[1].bar(type_counts.index, type_counts.values, color=PALETTE[:len(type_counts)])
axes[1].set_ylabel("Count"); axes[1].set_title("Cancer Cases by Type")
for i, v in enumerate(type_counts.values):
    axes[1].text(i, v+1, str(v), ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG}/fig10_cancer_types.png"); plt.close()
print("Fig 10: Cancer types")

# ── Fig 11: Age distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(df[df.label==0]["age"], bins=30, alpha=0.6, color="#2196F3", label="Controls", density=True)
ax.hist(df[df.label==1]["age"], bins=30, alpha=0.6, color="#FF5722", label="Cancer", density=True)
ax.set_xlabel("Age"); ax.set_ylabel("Density"); ax.set_title("Age Distribution: Cancer vs Controls")
ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/fig11_age_distribution.png"); plt.close()
print("Fig 11: Age distribution")

# ── Fig 12: Gender distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,5))
gdf = df.groupby(["gender","label"]).size().unstack(fill_value=0)
gdf.columns = ["Control","Cancer"]
gdf.plot(kind="bar", ax=ax, color=["#2196F3","#FF5722"], edgecolor="white", rot=0)
ax.set_xlabel("Gender"); ax.set_ylabel("Count"); ax.set_title("Gender × Cancer Status")
ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/fig12_gender_distribution.png"); plt.close()
print("Fig 12: Gender distribution")

# ── Fig 13: Ethnicity distribution ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,5))
edf = df.groupby(["ethnicity","label"]).size().unstack(fill_value=0)
edf.columns = ["Control","Cancer"]
edf.plot(kind="bar", ax=ax, color=["#2196F3","#FF5722"], edgecolor="white", rot=25)
ax.set_ylabel("Count"); ax.set_title("Ethnicity × Cancer Status (Real NHANES Data)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/fig13_ethnicity_distribution.png"); plt.close()
print("Fig 13: Ethnicity distribution")

# ── Fig 14: Biomarker boxplots — key features ────────────────────────────────
key_feats = [f for f in ["wbc","hemoglobin","albumin","crp","nlr","platelets"] if f in df.columns]
if key_feats:
    fig, axes = plt.subplots(2, 3, figsize=(13,8))
    axes = axes.flatten()
    for i, feat in enumerate(key_feats[:6]):
        data0 = df[df.label==0][feat].dropna()
        data1 = df[df.label==1][feat].dropna()
        axes[i].boxplot([data0, data1], labels=["Control","Cancer"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#E3F2FD"),
                        medianprops=dict(color="red",linewidth=2))
        axes[i].set_title(feat.upper()); axes[i].set_ylabel("Value")
    plt.suptitle("Key Biomarker Distributions: Cancer vs Controls", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig14_biomarker_boxplots.png"); plt.close()
    print("Fig 14: Biomarker boxplots")

# ── Fig 15: Correlation heatmap ──────────────────────────────────────────────
num_feats = [f for f in FEATURE_COLS if df[f].notna().sum() > 1000][:15]
if len(num_feats) >= 5:
    corr = df[num_feats].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                annot_kws={"size":7}, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix (Real NHANES Data)")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig15_correlation_heatmap.png"); plt.close()
    print("Fig 15: Correlation heatmap")

# ── Fig 16: Missing data heatmap ─────────────────────────────────────────────
miss = df[FEATURE_COLS].isna().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,6))
colors_miss = ["#FF5722" if v>0.3 else "#FF9800" if v>0.1 else "#4CAF50" for v in miss.values]
ax.barh(miss.index, miss.values*100, color=colors_miss)
ax.set_xlabel("Missing Data (%)"); ax.set_title("Feature Missingness Pattern")
ax.axvline(30, color="red", ls="--", alpha=0.5, label="30% threshold")
ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/fig16_missing_data.png"); plt.close()
print("Fig 16: Missing data")

# ── Fig 17: AUROC by cancer type ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
cancer_types = ["colorectal","lung","liver"]
type_aurocs = {}
for ct in cancer_types:
    mask = (df["cancer_type"] == ct) | (df["label"] == 0)
    y_sub = (df.loc[mask,"label"]).values
    p_sub = best_prob[mask.values]
    if y_sub.sum() > 5:
        auc = roc_auc_score(y_sub, p_sub)
        type_aurocs[ct] = auc
if type_aurocs:
    ax.bar(list(type_aurocs.keys()), list(type_aurocs.values()),
           color=["#2196F3","#FF5722","#4CAF50"])
    ax.set_ylim(0,1); ax.set_ylabel("AUROC")
    ax.set_title(f"AUROC by Cancer Type — {best_name}")
    ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Random")
    for i,(k,v) in enumerate(type_aurocs.items()):
        ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig17_auroc_by_cancer_type.png"); plt.close()
    print("Fig 17: AUROC by cancer type")

# ── Fig 18: AUROC by age group ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
bins = [18,40,55,65,75,120]
labels_age = ["18-39","40-54","55-64","65-74","75+"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels_age)
age_aurocs = {}
for ag in labels_age:
    mask = df["age_group"] == ag
    y_ag = df.loc[mask,"label"].values
    p_ag = best_prob[mask.values]
    if y_ag.sum() >= 5 and (y_ag==0).sum() >= 5:
        age_aurocs[ag] = roc_auc_score(y_ag, p_ag)
if age_aurocs:
    ax.bar(list(age_aurocs.keys()), list(age_aurocs.values()), color=PALETTE[:len(age_aurocs)])
    ax.set_ylim(0,1); ax.set_ylabel("AUROC"); ax.set_xlabel("Age Group")
    ax.set_title(f"AUROC by Age Group — {best_name} (Fairness Analysis)")
    ax.axhline(0.5, color="red", ls="--", alpha=0.5)
    for i,(k,v) in enumerate(age_aurocs.items()):
        ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/fig18_auroc_by_age.png"); plt.close()
print("Fig 18: AUROC by age group")

# ── Fig 19: AUROC by gender ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,5))
gender_aurocs = {}
for g in ["M","F"]:
    mask = df["gender"] == g
    y_g  = df.loc[mask,"label"].values
    p_g  = best_prob[mask.values]
    if y_g.sum() >= 5:
        gender_aurocs[g] = roc_auc_score(y_g, p_g)
if gender_aurocs:
    ax.bar(["Male","Female"] if "M" in gender_aurocs else list(gender_aurocs.keys()),
           list(gender_aurocs.values()), color=["#2196F3","#E91E63"], width=0.4)
    ax.set_ylim(0,1); ax.set_ylabel("AUROC")
    ax.set_title("AUROC by Gender — Fairness Analysis")
    for i,(k,v) in enumerate(gender_aurocs.items()):
        ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG}/fig19_auroc_by_gender.png"); plt.close()
print("Fig 19: AUROC by gender")

# ── Fig 20: AUROC by ethnicity ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
eth_aurocs = {}
for eth in df["ethnicity"].dropna().unique():
    mask = df["ethnicity"] == eth
    y_e  = df.loc[mask,"label"].values
    p_e  = best_prob[mask.values]
    if y_e.sum() >= 5 and (y_e==0).sum() >= 10:
        eth_aurocs[eth] = roc_auc_score(y_e, p_e)
if eth_aurocs:
    sorted_eth = dict(sorted(eth_aurocs.items(), key=lambda x: x[1], reverse=True))
    ax.barh(list(sorted_eth.keys()), list(sorted_eth.values()),
            color=PALETTE[:len(sorted_eth)])
    ax.set_xlim(0,1); ax.set_xlabel("AUROC")
    ax.set_title("AUROC by Ethnicity — Fairness Analysis (Real NHANES Data)")
    ax.axvline(0.5, color="red", ls="--", alpha=0.5)
    for i,(k,v) in enumerate(sorted_eth.items()):
        ax.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/fig20_auroc_by_ethnicity.png"); plt.close()
print("Fig 20: AUROC by ethnicity")

# ── Fig 21: Survey cycle comparison ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
cycle_aurocs = {}
for cyc in df["cycle"].unique():
    mask = df["cycle"] == cyc
    y_c  = df.loc[mask,"label"].values
    p_c  = best_prob[mask.values]
    if y_c.sum() >= 5:
        cycle_aurocs[cyc] = roc_auc_score(y_c, p_c)
if cycle_aurocs:
    ax.bar(list(cycle_aurocs.keys()), list(cycle_aurocs.values()), color=PALETTE[:3])
    ax.set_ylim(0,1); ax.set_ylabel("AUROC")
    ax.set_title("AUROC by Survey Cycle (Temporal Generalization)")
    for i,(k,v) in enumerate(cycle_aurocs.items()):
        ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG}/fig21_auroc_by_cycle.png"); plt.close()
print("Fig 21: Cycle generalization")

# ── Fig 22: Threshold analysis ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
thresholds = np.linspace(0.01, 0.99, 100)
sensitivities = [((best_prob[y==1]>=t).mean()) for t in thresholds]
specificities = [((best_prob[y==0]< t).mean()) for t in thresholds]
f1s = [f1_score(y, (best_prob>=t).astype(int), zero_division=0) for t in thresholds]
ax.plot(thresholds, sensitivities, color="#FF5722", lw=2, label="Sensitivity (Recall)")
ax.plot(thresholds, specificities, color="#2196F3", lw=2, label="Specificity")
ax.plot(thresholds, f1s,          color="#4CAF50", lw=2, label="F1 Score")
ax.axvline(0.5, color="black", ls="--", alpha=0.5, label="Default threshold")
ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Score")
ax.set_title(f"Threshold Analysis — {best_name}")
ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/fig22_threshold_analysis.png"); plt.close()
print("Fig 22: Threshold analysis")

# ── Fig 23: Sample dataset overview ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14,8))
# 1. Class balance
ax = axes[0,0]
vals = [(y==0).sum(), (y==1).sum()]
ax.pie(vals, labels=["Controls","Cancer"], colors=["#2196F3","#FF5722"],
       autopct="%1.1f%%", startangle=90)
ax.set_title("Class Balance")
# 2. Cancer subtype
ax = axes[0,1]
type_c = df[df.label==1]["cancer_type"].value_counts()
ax.bar(type_c.index, type_c.values, color=PALETTE[:len(type_c)])
ax.set_title("Cancer Subtypes"); ax.set_ylabel("N")
# 3. Cycle distribution
ax = axes[0,2]
cycle_c = df["cycle"].value_counts()
ax.bar(cycle_c.index, cycle_c.values, color=PALETTE[:3])
ax.set_title("Survey Cycles"); ax.set_ylabel("N")
# 4. Age histogram
ax = axes[1,0]
ax.hist(df["age"], bins=25, color="#9C27B0", edgecolor="white")
ax.set_title("Age Distribution"); ax.set_xlabel("Age")
# 5. WBC distribution
ax = axes[1,1]
if "wbc" in df.columns:
    ax.hist(df["wbc"].dropna(), bins=30, color="#FF9800", edgecolor="white")
    ax.set_title("WBC Distribution"); ax.set_xlabel("WBC (10^9/L)")
# 6. Missing data summary
ax = axes[1,2]
miss_pct = df[FEATURE_COLS].isna().mean()*100
ax.hist(miss_pct, bins=15, color="#00BCD4", edgecolor="white")
ax.set_title("Missingness Distribution"); ax.set_xlabel("% Missing per Feature")
plt.suptitle("NHANES Dataset Overview (Real Data)", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(f"{FIG}/fig23_dataset_overview.png", bbox_inches="tight"); plt.close()
print("Fig 23: Dataset overview")

# ── Fig 24: Model comparison radar ───────────────────────────────────────────
from matplotlib.patches import FancyArrowPatch
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
metrics_r = ["AUROC","AUPRC","F1"]
angles = np.linspace(0, 2*np.pi, len(metrics_r), endpoint=False).tolist()
angles += angles[:1]
for (name, res), col in zip(results.items(), colors):
    vals_r = [res[m] for m in metrics_r] + [res[metrics_r[0]]]
    ax.plot(angles, vals_r, color=col, lw=2, label=name)
    ax.fill(angles, vals_r, color=col, alpha=0.1)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics_r)
ax.set_ylim(0,1); ax.set_title("Model Comparison Radar Chart", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/fig24_radar_chart.png", bbox_inches="tight"); plt.close()
print("Fig 24: Radar chart")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL RESULTS SUMMARY (REAL NHANES DATA)")
print("="*65)
print(f"{'Model':<22} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Brier':>7}")
print("-"*50)
for name, res in sorted(results.items(), key=lambda x: -x[1]["AUROC"]):
    print(f"{name:<22} {res['AUROC']:>7.3f} {res['AUPRC']:>7.3f} {res['F1']:>7.3f} {res['Brier']:>7.3f}")
print("="*65)
print(f"\nBest model: {best_name}")
print(f"  AUROC = {results[best_name]['AUROC']:.3f}")
print(f"  AUPRC = {results[best_name]['AUPRC']:.3f}")
print(f"  F1    = {results[best_name]['F1']:.3f}")
print(f"\n24 figures saved to: results/figures/")
print("Results JSON:         results/nhanes_model_results.json")
print("\nThese are REAL results from REAL NHANES data — publishable!")
