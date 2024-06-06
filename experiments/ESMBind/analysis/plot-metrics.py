# basic
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
# torch tools
import torch
from loguru import logger
# metrics
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

# ==================== Configuration ====================
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "fine-tuned-esmbind-inference-performance" / "r2"
LABELS = BASE / "fine-tune-input" / "labels.pkl"

# ==================== Function ====================


# ==================== Main ====================
# 0. load results
with open(RESULTS / "result.pkl", "rb") as f:
    results = pickle.load(f)
# 1. load abdbids and labels
with open(LABELS.parent / "abdbids.txt", "r") as f:
    abdbids = [line.strip() for line in f.readlines()]

with open(LABELS, "rb") as f:
    labels = pickle.load(f)

all = {"abdbid": [], "mcc": [], "aucroc": [], "precision": [], "recall": [], "f1": []}

# 2. calculate mcc metrics
warn_items_num = 0
warn_items = {"abdbid": [], "idx": []}
for i, (abdbid, x, y) in enumerate(zip(abdbids, labels, results)):
    if len(x) == len(y["preds"]):
        all["abdbid"].append(abdbid)
        all["mcc"].append(matthews_corrcoef(x, y["preds"]))
        all["aucroc"].append(roc_auc_score(x, y["preds"]))
        all["precision"].append(precision_score(x, y["preds"]))
        all["recall"].append(recall_score(x, y["preds"]))
        all["f1"].append(f1_score(x, y["preds"]))
    else:
        # warning that a sample may have exceeded the max length the model can accept
        logger.warning(
            f"Warning: a sample has length {len(x)} that exceeds the max length 1024 the model can accept:\n{y['raw_seq']}\n"
        )
        warn_items_num += 1
        warn_items["idx"].append(i)
        warn_items["abdbid"].append(abdbid)
print(warn_items_num)  # 145

# write warn_items to file json
with open("warn_items.json", "w") as f:
    json.dump(warn_items, f)

# ----------------------------------------
# Test set performance
# ----------------------------------------
split_dict = torch.load("/mnt/bob/shared/AsEP/split/split_dict.pt")
split_idx = split_dict["epitope_group"]
with open("/mnt/bob/shared/AsEP/raw/asepv1-AbDb-IDs.txt") as f:
    asepv1_abdbids = [line.strip() for line in f.readlines()]

test_set_abdbids = np.array(asepv1_abdbids)[split_idx["test"].tolist()]
val_set_abdbids = np.array(asepv1_abdbids)[split_idx["val"].tolist()]
train_set_abdbids = np.array(asepv1_abdbids)[split_idx["train"].tolist()]
abdbid2set = {}
for abdbid in abdbids:
    if abdbid in test_set_abdbids:
        abdbid2set[abdbid] = "test"
    elif abdbid in val_set_abdbids:
        abdbid2set[abdbid] = "val"
    elif abdbid in train_set_abdbids:
        abdbid2set[abdbid] = "train"
    else:
        print(f"abdbid {abdbid} not found in any set")


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
