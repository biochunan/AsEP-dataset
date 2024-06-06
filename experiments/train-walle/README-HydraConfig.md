# Hydra Config

This document describes the Hydra configuration for the retraining of the WALLE model, with a general overview of the configuration files and their usage.

Refer to the section [General Usage of Hydra Config](#general-usage-of-hydra-config-skip-this-if-you-are-familiar-with-hydra) for the general usage of Hydra configuration.

<!-- insert a table of contents -->
- [Configuration Files](#configuration-files)
  - [Install cli autocomplete for hydra](#install-cli-autocomplete-for-hydra)
  - [`config.yaml`](#configyaml)
  - [`callbacks/default_callbacks.yaml`](#callbacksdefault_callbacksyaml)
  - [`dataset/dataset.yaml`](#datasetdatasetyaml)
  - [`hparams/hparams.yaml`](#hparamshparamsyaml)
  - [`hydra/hydra.yaml`](#hydrahydrayaml)
  - [`loss/loss.yaml`](#losslossyaml)
  - [`optimizer/adam.yaml`](#optimizeradamyaml)
  - [`wandb_init/wandb.yaml`](#wandb_initwandbyaml)
- [General Usage of Hydra Config (skip this if you are familiar with Hydra)](#general-usage-of-hydra-config-skip-this-if-you-are-familiar-with-hydra)

## Configuration Files

The configuration files are organized in the following structure:

```shell
conf/
├── config.yaml
├── callbacks
│   └── default_callbacks.yaml
├── dataset
│   └── dataset.yaml
├── hparams
│   └── hparams.yaml
├── hydra
│   └── hydra.yaml
├── loss
│   └── loss.yaml
├── optimizer
│   └── adam.yaml
└── wandb_init
    └── wandb.yaml
```

### Install cli autocomplete for hydra

```shell
eval "$(python train.py -sc install=zsh)"
```

### `config.yaml`

This is the top level configuration file that imports other configuration files.
It is used to define the configuration schema and the default values for the model training.

### `callbacks/default_callbacks.yaml`

This file contains the configuration for the default callbacks used in the model training.

```yaml
early_stopping:
    patience: 10
    min_delta: 0.0
    minimize: false
    metric_name: "valEpoch/avg_epi_node_auprc"

model_checkpoint:
    save_dir:  ${hydra:runtime.cwd}/ckpts
    k: 3
    minimize: false
    metric_name: "valEpoch/avg_epi_node_auprc"

# ------------------------------
# learning rate scheduler
# ------------------------------
lr_scheduler: null
```

- default learning rate scheduler is set to `null`. But you can use one of the following learning rate schedulers
  - [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
    ```yaml
    lr_scheduler:
        name: "ReduceLROnPlateau"
        kwargs:
            mode: "max"
            factor: 0.1
        step:
            metrics: "valEpoch/avg_epi_node_auprc"
    ```
  - [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
    ```yaml
    lr_scheduler:
        name: "StepLR"
        kwargs:
            step_size: 10
            gamma: 0.9
        step: null  # required by ReduceLROnPlateau, see example above
    ```


### `dataset/dataset.yaml`

```yaml
root: /mnt/Data            # root path for storing AsEP dataset
name: AsEP                 # dataset name
node_feat_type: 'pre_cal'  # choices=['pre_cal', 'one_hot', 'custom']

ab:
  embedding_model: 'igfold'     # choices: ['igfold', 'esm2', 'one_hot'， 'custom']
  custom_embedding_method: null # Optional[Callable]
  custom_embedding_method_src:
    script_path: null           # Optional[str]
    method_name: null           # Optional[str]

ag:
  embedding_model: 'esm2'       # choices: ['esm2', 'one_hot'， 'custom']
  custom_embedding_method: null # Optional[Callable]
  custom_embedding_method_src:
    script_path: null           # Optional[str]
    method_name: null           # Optional[str]

split_idx: null  # choices: [null, epitope_ratio, epitope_group]
                 # if null, default to epitope_ratio
```

<!-- TODO: add documentation -->

### `hparams/hparams.yaml`

This file contains the hyperparameters configuration for the model training.

```yaml
train_batch_size: 32
val_batch_size: 32
test_batch_size: 32

batch_size: ${.train_batch_size}
max_epochs: 100
pos_weight: 100.

input_ab_dim: 512
input_ag_dim: 480
input_ab_act: "relu"
input_ag_act: "relu"
dim_list:
  - 128
  - 64
act_list:
  - null

decoder:
  name: 'inner_prod'
num_edge_cutoff: 3
edge_cutoff: 0.5
```

- To change the model architecture, e.g. adding more layers:

    ```yaml
    dim_list:
      - 128
      - 128
      - 64
    act_list:
      - "relu"
      - null
    ```
    - the number of activation functions in `act_list` should be one less than the number of layers in `dim_list`. This is because the last layer is directly connected to the decoder; the activation function for the input layer is set separately in `input_ab_act` and `input_ag_act`.

- the default `decoder` is set to inner product `inner_prod`; To use a different `decoder`:
    - `fc` short for fully connected layer
        ```yaml
        decoder:
            # fc
            name: "fc"  # inner_prod, fc
            bias: true
            dropout: 0.1  # either null or float in (0, 1)
        ```
    - `bilinear` short for bilinear layer
        ```yaml
        decoder:
            # bilinear
            name: 'bilinear'
            init_method: 'xavier_uniform_'
        ```

### `hydra/hydra.yaml`

<!-- TODO: Add documentation -->

### `loss/loss.yaml`

<!-- TODO: Add documentation -->

### `optimizer/adam.yaml`

<!-- TODO: Add documentation -->

###  `wandb_init/wandb.yaml`

<!-- TODO: Add documentation -->

## General Usage of Hydra Config (skip this if you are familiar with Hydra)

The folder `./conf` contains the hydra parameters configuration, which serves as a template for the model training. The configuration files are written in `YAML` format, and the parameters are organized in a hierarchical structure. The configuration files are used to define the hyperparameters for the model training, and they can be overridden from the command line.

To run the model training with the default configuration or overrides parameters from the command line, you can use the following command:

```shell
# run with default configuration
python train.py

# this overrides the parameter `hparams.train_batch_size=64` in the default configuration
python train.py hparams.train_batch_size=64
```

