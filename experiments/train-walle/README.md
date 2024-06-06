# Training notes

## Sweep - BLOSUM62 embedding

```shell
wandb sweep ./sweep-config/sweep-walle-edge-level-blosum62.yaml
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id>
```

## Sweep - one hot embedding

```shell
wandb sweep ./sweep-config/sweep-walle-edge-level-onehot.yaml
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
```

##  Sweep - linear model (i.e. without graph component)

```shell
wandb sweep ./sweep-config/linear-model-edge-level.yaml
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
```