import json, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
with open('results/agent_results_100.json') as f:
    recs = json.load(f)
n = len(recs)
eas_j  = [r['eas_jaccard'] for r in recs]
eas_o  = [r['eas_overlap_k'] for r in recs]
hall   = [r['hallucination_rate'] for r in recs]
modes  = [r.get('mode','5agent') for r in recs]
from collections import Counter
print(f'Progress: {n}/100 done')
print(f'EAS-J:  {np.mean(eas_j):.4f} +/- {np.std(eas_j):.4f}')
print(f'EAS-O5: {np.mean(eas_o):.4f} +/- {np.std(eas_o):.4f}')
print(f'Hall:   {np.mean(hall):.4f} +/- {np.std(hall):.4f}')
print(f'Modes:  {dict(Counter(modes))}')
ct_key = 'cancer_type'
ct_counts = Counter(r[ct_key] for r in recs)
print(f'Types:  {dict(ct_counts)}')
last = recs[-1]
print(f'Last:   idx={last["eval_idx"]} ctype={last[ct_key]} EAS={last["eas_jaccard"]} Hall={last["hallucination_rate"]}')
