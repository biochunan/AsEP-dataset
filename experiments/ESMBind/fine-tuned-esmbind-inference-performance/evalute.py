# basic
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE = Path(__file__).resolve().parent
seqres = BASE/'assets'/'seqres.pkl'
results = BASE/'r2'/'result.pkl'
labels = BASE/'assets'/'labels.pkl'
abdbids = BASE/'assets'/'abdbids.txt'

with open(abdbids, 'r') as f:
    abdbids = f.read().splitlines()

with open(seqres, 'rb') as f:
    seqres = pickle.load(f)

with open(results, 'rb') as f:
    esmbind_results = pickle.load(f)

with open(labels, 'rb') as f:
    labels = pickle.load(f)

# 2. calculate mcc metrics
esmbind_metrics = {'abdbid': [], 'mcc': [], 'precision': [], 'recall': [], 'aucroc': [], 'f1': []}
warn_items = 0
warn_indices = []
for i, (n, x, y) in enumerate(zip(abdbids, labels, esmbind_results)):
    if len(x) == len(y['preds']):
        esmbind_metrics['abdbid'].append(n)
        esmbind_metrics['mcc'].append(matthews_corrcoef(x, y['preds']))
        esmbind_metrics['precision'].append(precision_score(x, y['preds']))
        esmbind_metrics['recall'].append(recall_score(x, y['preds']))
        esmbind_metrics['aucroc'].append(roc_auc_score(x, y['preds']))
        esmbind_metrics['f1'].append(f1_score(x, y['preds']))
    else:
        # warning that a sample may have exceeded the max length the model can accept
        print(f"Warning: a sample has length {len(x)} that exceeds the max length 1024 the model can accept:\n{y['raw_seq']}\n")
        warn_items += 1
        warn_indices.append(i)
print(warn_items)  # 145

pd.DataFrame(esmbind_metrics).to_csv('esmbind_metrics.csv', index=False)
