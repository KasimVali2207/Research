"""Patch full_results_summary.json — add 95CI EAS field that was missing."""
import json, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")

with open("results/agent_results_100.json") as f:
    recs = json.load(f)
eas_j = [r["eas_jaccard"] for r in recs]
n = len(eas_j)
ci = [round(np.mean(eas_j) - 1.96*np.std(eas_j)/np.sqrt(n), 4),
      round(np.mean(eas_j) + 1.96*np.std(eas_j)/np.sqrt(n), 4)]

with open("results/full_results_summary.json") as f:
    s = json.load(f)
s["agentic"]["95ci_eas_jaccard"] = ci
s["agentic"]["n_patients_run"] = n
with open("results/full_results_summary.json","w") as f:
    json.dump(s, f, indent=2)
print(f"Fixed: 95ci_eas_jaccard = {ci}  (n={n})")
