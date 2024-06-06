# AsEP Dataset

Antibody-specific Epitope Prediction dataset

## Installation

```shell
# enable conda (init zsh if you are using zsh, or init bash etc.)
conda init zsh

# you can also use `make` to prepare the conda environment
make setup-gpu-env

# if you don't have a GPU, then run
# make setup-cpu-env

# install other dependencies
make install-dependencies
```

This requires `make`, run `sudo apt install make` to install.

This will do the following:

- create a conda environment named `walle`
- install the required packages
- install the `asep` package in editable mode.

## Download dataset

We provide console scripts to download the dataset. You can download the dataset by running the following command:

```shell
download-asep /path/to/directory AsEP
```

- `/path/to/directory` is the directory where you want to save the dataset.
- `AsEP` is the name of the dataset, by default, it is `AsEP`.

## Package `AsEP`

The package `asep` provides the following functionalities:

- Download the dataset (see above)
- Provides code for constructing the neural network (Protein Language Model and Graph Representation of antibody-antigen complexes) proposed in the manuscript
- Provides code for training the model
- Provides code for evaluating the model
