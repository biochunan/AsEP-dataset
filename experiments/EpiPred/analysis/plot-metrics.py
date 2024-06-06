# basic
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("Set2")

# torch tools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts

# ==================== Configuration ====================
BASE = Path(__file__).resolve().parents[1]
METRICS = BASE / "metrics"


# ==================== Function ====================
def get_top1_metric(json_fp: str) -> Tuple[str, str, float]:
    with open(json_fp, "r") as f:
        data = json.load(f)
    d = data["rank1"]
    d["abdbid"] = data["abdbid"]
    return d


# ==================== Main ====================
# get top-1 prediction from each json
json_fps = [str(f) for f in METRICS.glob("*.json")]
all = {"abdbid": [], "mcc": [], "aucroc": [], "precision": [], "recall": [], "f1": []}

for fp in json_fps:
    try:
        d = get_top1_metric(fp)
    except Exception as e:
        print(f"Error in {fp}: {e}")
        continue
    for k, v in get_top1_metric(fp).items():
        all[k].append(v)

# plot metrics as histogram
fig, ax = plt.subplots(2, 3, figsize=(9, 6))  # , sharex=True, sharey=True)
# create bin between -1 and 1 using np
# sns.histplot(data=all, x="mcc", bins=np.linspace(-1, 1, 21), ax=ax)
for i, k in enumerate(["mcc", "aucroc", "precision", "recall", "f1"]):
    sns.histplot(data=all, x=k, bins=20, ax=ax[i // 3][i % 3])
    ax[i // 3][i % 3].set_title(k)
    # ax[i//3][i%3].tick_params(axis="x", bottom=True, labelbottom=True)
# set super title
fig.suptitle("Metrics on 1723 AbAg Samples")
# remove the last plot
ax[1][2].remove()
fig.tight_layout()
fig.show()
# fig.savefig('plots/metrics-hist.pdf', dpi=300)

# violin plot
fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True)
for i, k in enumerate(["mcc", "aucroc", "precision", "recall", "f1"]):
    sns.violinplot(data=all, x=k, ax=ax[i // 3][i % 3])
    ax[i // 3][i % 3].set_title(k)
    # show xticks
    ax[i // 3][i % 3].tick_params(axis="x", bottom=True, labelbottom=True)
# set super title
fig.suptitle("Metrics on 1723 AbAg Samples")
# remove the last plot
ax[1][2].remove()
fig.tight_layout()
fig.show()
# fig.savefig('plots/metrics-violine.pdf', dpi=300)

# ----------------------------------------
# Test set performance
# ----------------------------------------
split_dict = torch.load("/mnt/bob/shared/AsEP/split/split_dict.pt")
split_idx = split_dict['epitope_group']
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
        # raise ValueError(f"abdbid {abdbid} not found in any set")
        print(f"abdbid {abdbid} not found in any set")

# derive metrics from all using abdbids
all_df = pd.DataFrame(all)
all_df['set'] = all_df['abdbid'].map(abdbid2set)

test_set_metric_df = all_df[all_df['set'] == 'test']

# get a summary of each metric
summary = test_set_metric_df.describe()
# add stderr
summary.loc['stderr'] = test_set_metric_df.sem()
# save to csv
summary.to_csv('metrics-summary.csv')

# print summary with 3 decimals
print(summary.round(3))


# plot metrics as histogram
fig, ax = plt.subplots(2, 3, figsize=(9, 6))#, sharex=True, sharey=True)
# create bin between -1 and 1 using np
for i, k in enumerate(['mcc', 'aucroc', 'precision', 'recall', 'f1']):
    sns.histplot(data=test_set_metric_df, x=k, bins=20, ax=ax[i//3][i%3])
    ax[i//3][i%3].set_title(k)
    # ax[i//3][i%3].tick_params(axis="x", bottom=True, labelbottom=True)
# remove the last plot
ax[1][2].remove()
# set super title
fig.suptitle("Test Set Metrics")
# other settings
fig.tight_layout()
fig.show()
# fig.savefig('plots/metrics-hist-testset.pdf', dpi=300)
