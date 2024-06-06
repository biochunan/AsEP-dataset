# basic
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
# torch tools
# torch tools
import torch

# other
# metrics

# ==================== Configuration ====================
BASE=Path(__file__).resolve().parents[1]
METRICS=BASE/'metrics'

# ==================== Function ====================
def get_metric(json_fp: str) -> Tuple[str, str, float]:
    with open(json_fp, 'r') as f:
        data = json.load(f)
    d = {'abdbid': data['abdbid'],
         'mcc': data['mcc'],
         'aucroc': data['aucroc'],
         'precision': data['precision'],
         'recall': data['recall'],
         'f1': data['f1']}
    return d

# ==================== Main ====================
# get top-1 prediction from each json
json_fps = [str(f) for f in METRICS.glob('*.json')]
all = {'abdbid': [], 'mcc': [], 'aucroc': [], 'precision': [], 'recall': [], 'f1': []}

for fp in json_fps:
    for k, v in get_metric(fp).items():
        all[k].append(v)
len(all['abdbid'])

# ----------------------------------------
# test set
# ----------------------------------------
split_dict = torch.load("/mnt/bob/shared/AsEP/split/split_dict.pt")
split_idx = split_dict["epitope_group"]
with open("/mnt/bob/shared/AsEP/raw/asepv1-AbDb-IDs.txt") as f:
    asepv1_abdbids = [line.strip() for line in f.readlines()]

abdbids = all['abdbid']
test_set_abdbids  = np.array(asepv1_abdbids)[split_idx['test'].tolist()]
val_set_abdbids   = np.array(asepv1_abdbids)[split_idx['val'].tolist()]
train_set_abdbids = np.array(asepv1_abdbids)[split_idx['train'].tolist()]
abdbid2set = {}
for abdbid in abdbids:
    if abdbid in test_set_abdbids:
        abdbid2set[abdbid] = 'test'
    elif abdbid in val_set_abdbids:
        abdbid2set[abdbid] = 'val'
    elif abdbid in train_set_abdbids:
        abdbid2set[abdbid] = 'train'
    else:
        print(f"abdbid {abdbid} not found in any set")


# derive metrics from all using abdbids
all_df = pd.DataFrame(all)
all_df['set'] = all_df['abdbid'].map(abdbid2set)

test_set_metric_df = all_df[all_df['set'] == 'test']


# derive metrics from all using abdbids
all_df = pd.DataFrame(all)
all_df["set"] = all_df["abdbid"].map(abdbid2set)

test_set_metric_df = all_df[all_df["set"] == "test"]

# get a summary of each metric
summary = test_set_metric_df.describe()
# add stderr
summary.loc["stderr"] = test_set_metric_df.sem()
# save to csv
summary.to_csv("metrics-summary.csv")

# print summary with 3 decimals
print(summary.round(3))
