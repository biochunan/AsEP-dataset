# Evaluate pre-trained model

Contents

- [Evaluate pre-trained model](#evaluate-pre-trained-model)
  - [Download ckpts from wandb](#download-ckpts-from-wandb)
  - [Evaluate ckpts](#evaluate-ckpts)
  - [Calculate mean metrics from evaluation results](#calculate-mean-metrics-from-evaluation-results)
  - [Evaluation procedures](#evaluation-procedures)
    - [Validate config](#validate-config)
  - [Evaluation Results Schema](#evaluation-results-schema)
    - [Config modification for old previous versions of config](#config-modification-for-old-previous-versions-of-config)
  - [Pre-trained model and config](#pre-trained-model-and-config)
    - [Split by epitope/surf ratio](#split-by-epitopesurf-ratio)
    - [Split by epitope group](#split-by-epitope-group)

## Download ckpts from wandb

```shell
python download-ckpts.py \
-c chunan/retrain-walle/config:v465  \
-b chunan/retrain-walle/best_k_models:v376 \
-o assets/whole-sweep-72
```

- `-o`: output directory to save the artifacts, the directory basename is best to be the same as the run name.
- `-c`: the config run name.
- `-b`: the best k models run name.

## Evaluate ckpts

```shell
python evaluate_on_walle.py \
-o ./metrics/crisp-sweep-1 \
-c ./assets/crisp-sweep-1/crisp-sweep-1.yaml \
-m ./assets/crisp-sweep-1/rank_0-epoch_148.pt \
> ./metrics/crisp-sweep-1.log 2>&1
```

- `-o`: output directory to save the metrics.
- `-c`: the config file.
- `-m`: the model file.

## Calculate mean metrics from evaluation results

```shell
python calculate_mean_metrics.py ./metrics/crisp-sweep-1 \
-json ./metrics/crisp-sweep-1-summary.json \
# --quiet
```

This script will calculate the mean metrics from all the evaluation results in the directory, e.g. `./metrics/crisp-sweep-1`.

- `-json`: save the summary to a json file.
- `--quiet`: do not print the summary to stdout.

## Evaluation procedures

### Validate config

For baseline models using custom BLOSUM62 matrix, you need to provide the correct path to the blosum62.py under the package `asep` which is located under the root of this repo `AsEP-dataset/asep/data/embedding/blosum62.py`.

In your `config.yaml`, correct the `script_path` under `dataset` section to the path to `blosum62.py` on your system.

NOTE: ignore the embedding_model field for the custom embedding method because they are ignored if both `method_name` and `script_path` are provided. You just need to change the `script_path` because `method_name` is already set to `embed_blosum62` which comes with the script.

```yaml
dataset:
  ab:
    custom_embedding_method: null
    custom_embedding_method_src:
      method_name: embed_blosum62
      script_path: /your/path/to/AsEP-dataset/asep/data/embedding/blosum62.py
    embedding_model: igfold
  ag:
    custom_embedding_method: null
    custom_embedding_method_src:
      method_name: embed_blosum62
      script_path: /your/path/to/AsEP-dataset/asep/data/embedding/blosum62.py
    embedding_model: esm2
  name: AsEP
  node_feat_type: custom
  root: /mnt/Data
  split_idx: null
```

## Evaluation Results Schema

For `calculate-mean-mcc.py`

```json
{
  {
    "node": {
      "mcc": {
        "mean": float,
        "stddev": float,
        "stderr": float,
        "metadata": {
            "n": int,  // number of samples
            "mcc": {"7wkx_1P": float, ...},  // sample performance
      },
      "auc_roc": {...},
      "precision": {...},
      "recall": {...},
      "f1": {...},

    },
  },
  "link": {
    "mcc": {...},
    "auc_roc": {...},
    "precision": {...},
    "recall": {...},
    "f1": {...},
  }
}
```

### Config modification for old previous versions of config

- may need to add a field `model_type:"graph"` to hparams field
- may need to add a field `split_method:null` to dataset field

## Pre-trained model and config

### Split by epitope/surf ratio

|    Run Name    | Model  | Embedding (Ab/Ag) | \# GCN Layers |
| :------------: | :----: | :---------------: | :-----------: |
| clear-sweep-30 | graph  |      onehot       |       2       |
| comic-sweep-9  | graph  |     ESM2/ESM2     |       2       |
| decent-sweep-2 | graph  |     BLOSUM62      |       2       |
| lilac-sweep-16 | linear |     ESM2/ESM2     |       2       |
|  sage-sweep-9  | linear |    IgFold/ESM2    |       2       |
| whole-sweep-72 | graph  |    IgFold/ESM2    |       2       |

Best performing model on `valEpoch/avg_edge_index_bg_mcc`:

|     Run Name      | Model | Embedding (Ab/Ag) | \# GCN Layers |
| :---------------: | :---: | :---------------: | :-----------: |
| treasured-sweep-8 | graph |    IgFold/ESM2    |       2       |


### Split by epitope group

|     Run Name     | Model | Embedding (Ab/Ag) | \# GCN layers |
| :--------------: | :---: | :---------------: | :-----------: |
| jumping-sweep-22 | graph |    IgFold/ESM2    |       2       |
|   wild-sweep-1   | graph |    IgFold/ESM2    |       3       |
