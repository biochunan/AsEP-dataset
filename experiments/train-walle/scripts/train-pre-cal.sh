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
    "wandb_init.notes='pre_cal'" \
    hparams.max_epochs=5 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='pre_cal' \
    dataset.ab.embedding_model='igfold' \
    dataset.ag.embedding_model='esm2'
fi

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
if [[ ${mode} == 'train' ]]; then
  python train.py \
    mode='train' \
    "wandb_init.project=retrain-walle" \
    "wandb_init.notes='pre_cal'" \
    "wandb_init.tags=['pre_cal']" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=32 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='pre_cal' \
    dataset.ab.embedding_model='igfold' \
    dataset.ag.embedding_model='esm2' \
    "callbacks.early_stopping=null"
fi