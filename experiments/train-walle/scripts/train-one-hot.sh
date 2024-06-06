#!/bin/zsh

mode=${1}

if [[ -z ${mode} ]]; then
  echo "mode is not provided, choices: dev, train"
  exit 1
fi

# ------------------------------------------------------------------------------
# dev
# ------------------------------------------------------------------------------
if [[ ${mode} == 'dev' ]]; then
  python train.py \
    mode='dev' \
    "wandb_init.project=retrain-walle-dev" \
    "wandb_init.notes='one_hot'" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='one_hot' \
    hparams.input_ab_dim=20 \
    hparams.input_ag_dim=20
fi

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
if [[ ${mode} == 'train' ]]; then
  python train.py \
    mode='train' \
    "wandb_init.project=retrain-walle" \
    "wandb_init.notes='one_hot'" \
    "wandb_init.tags=['one_hot']" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='one_hot' \
    hparams.input_ab_dim=20 \
    hparams.input_ag_dim=20 \
    "callbacks.early_stopping=null"
fi

# TODO: turn off early stopping