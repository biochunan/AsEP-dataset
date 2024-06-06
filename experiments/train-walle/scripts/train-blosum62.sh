#!/bin/zsh

mode=${1}

if [[ -z ${mode} ]]; then
  echo "mode is not provided"
  exit 1
fi

# ------------------------------------------------------------------------------
# dev
# ------------------------------------------------------------------------------
if [[ ${mode} == 'dev' ]]; then
  python train.py \
    mode='dev' \
    "wandb_init.project=retrain-walle-dev" \
    "wandb_init.notes='BLOSUM62'" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='custom' \
    dataset.ab.custom_embedding_method_src.script_path='/workspaces/AsEP-dataset/asep/data/embedding/blosum62.py' \
    dataset.ab.custom_embedding_method_src.method_name='embed_blosum62' \
    dataset.ag.custom_embedding_method_src.script_path='/workspaces/AsEP-dataset/asep/data/embedding/blosum62.py' \
    dataset.ag.custom_embedding_method_src.method_name='embed_blosum62' \
    hparams.input_ab_dim=24 \
    hparams.input_ag_dim=24
fi

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
if [[ ${mode} == 'train' ]]; then
  python train.py \
    mode='train' \
    "wandb_init.project=retrain-walle" \
    "wandb_init.notes='blosum62'" \
    "wandb_init.tags=['blosum62']" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='custom' \
    dataset.ab.custom_embedding_method_src.script_path='/workspaces/AsEP-dataset/asep/data/embedding/blosum62.py' \
    dataset.ab.custom_embedding_method_src.method_name='embed_blosum62' \
    dataset.ag.custom_embedding_method_src.script_path='/workspaces/AsEP-dataset/asep/data/embedding/blosum62.py' \
    dataset.ag.custom_embedding_method_src.method_name='embed_blosum62' \
    hparams.input_ab_dim=24 \
    hparams.input_ag_dim=24 \
    "callbacks.early_stopping=null"
fi