# ESMBind

Contents

- [ESMBind](#esmbind)
  - [Dependencies](#dependencies)
  - [Reproducing LoRA fine-tuning on downstream binding site dataset](#reproducing-lora-fine-tuning-on-downstream-binding-site-dataset)
    - [Fine tune ESM2 to derive LoRA model for binding site prediction](#fine-tune-esm2-to-derive-lora-model-for-binding-site-prediction)
  - [Fine-tuned model configuration](#fine-tuned-model-configuration)
  - [Protein sequences that exceed the max length](#protein-sequences-that-exceed-the-max-length)
  - [Inference on AsEP data](#inference-on-asep-data)
  - [Evaluation](#evaluation)

## Dependencies

Download the following files and decompress to the same folder as this README file, refer to section [Reproducing LoRA fine-tuning on downstream binding site dataset](#reproducing-lora-fine-tuning-on-downstream-binding-site-dataset) for more details on the files:
- [:link: `abag_dataset.tar.gz`](https://drive.google.com/open?id=1Tu34QHk0ADIJBv2JFYS0VcVazrWH6wpU&usp=drive_fs) `4.7GB` (after decompression `6.6GB`)
- [:link: `binding_sites_random_split_by_family_550K.tar.gz`](https://drive.google.com/open?id=1f8praQdac9QK-VVTfvIgyd87JqyY8tXe&usp=drive_fs) `486MB` (after decompression `4.2GB`)

## Reproducing LoRA fine-tuning on downstream binding site dataset

> Credits to [@AmelieSchreiber](https://huggingface.co/AmelieSchreiber)
> - [AmelieSchreiber/ESMBind](https://huggingface.co/blog/AmelieSchreiber/esmbind) for the instructions.
> - [AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp3](https://huggingface.co/AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp3/tree/main) for the fine-tuning script.


This method fine-tunes PLM `esm2_t12_35M_UR50D` model on a downstream binding site dataset.

See this blog and repo for more details

Blog: [:link: AmelieSchreiber/esmbind](https://huggingface.co/blog/)AmelieSchreiber/esmbind

Code: [:link: esm2_t12_35M_lora_binding_sites_v2_cp3](https://huggingface.co/AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp3)

Dataset: [:link: binding_sites_random_split_by_family_550K](https://huggingface.co/datasets/AmelieSchreiber/binding_sites_random_split_by_family_550K)

Sub-folders:

- `abag_dataset`
  - `fine-tune-input`: contains the input data for evaluating the fine-tuned model (antigens derived from `AsEP`)
  - `seqres2interface`: these are mask tensor files, each mask is a binary 0/1 indicating non-binding-site and binding-site for each antigen residue
  - downloadable from [:link: abag_dataset.tar.gz](https://drive.google.com/open?id=1Tu34QHk0ADIJBv2JFYS0VcVazrWH6wpU&usp=drive_fs)
- `fine-tuned-esmbind-inference-performance`: contains the performance of this fine-tuned model on WALLE1.0 antigen dataset
  - `r2`: performance of the fine-tuned model with `r=2` i.e. `lora_binding_sites/best_model_esm2_t12_35M_lora_2023-11-09_12-02-07`
  - downloadable from [:link: fine-tuned-esmbind-inference-performance.tar.gz](https://drive.google.com/open?id=148Av_Sr4ZCI77qYrgiFRgS1WQzoU5Rnm&usp=drive_fs)
- `binding_sites_random_split_by_family_550K`: contains the raw input data for fine-tuning ESM2 to derive the LoRA model (ESMBind) for binding site prediction trained on general protein sequences
  - downloadable from [:link: binding_sites_random_split_by_family_550K.tar.gz](https://drive.google.com/open?id=1f8praQdac9QK-VVTfvIgyd87JqyY8tXe&usp=drive_fs)
- `lora_binding_sites`: contains the fine-tuned model checkpoints

### Fine tune ESM2 to derive LoRA model for binding site prediction

First download the dataset `binding_sites_random_split_by_family_550K` and decompress it to the same folder as this README file.

We provide the script `fine-tune-lora-esm2.py` to fine-tune the ESM2 model to derive the LoRA model for binding site prediction.


## Fine-tuned model configuration

We provided the fine-tuned model in `lora_binding_sites/best_model_esm2_t12_35M_lora_2023-11-09_12-02-07`

```python
# Set the LoRA config
config = {
    "r": 2,
    "lora_alpha": 1, #try 0.5, 1, 2, ..., 16
    "lora_dropout": 0.2,
    "lr": 5.701568055793089e-04,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.5,
    "num_train_epochs": 3,  # 3, 0.001
    "per_device_train_batch_size": 12,
    "weight_decay": 0.2,
    # Add other hyperparameters as needed
}
# The base model you will train a LoRA on top of
model_checkpoint = "facebook/esm2_t12_35M_UR50D"
```

## Protein sequences that exceed the max length

Entries that exceed the max length accepted by the model were saved in `warn_items.json`, this is due to esmbind can only take protein sequences of length up to 1024 residues as input:

```shell
2024-05-17 19:57:01.193 | WARNING  | __main__:<module>:77 - Warning: a sample has length 1130 that exceeds the max length 1024 the model can accept:
```

## Inference on AsEP data

We provide a python script `fine-tuned-esmbind-inference-performance/inference-finetuned-lora-esm2.py` to run inference on the AsEP dataset using the best fine-tuned model provided in `lora_binding_sites/best_model_esm2_t12_35M_lora_2023-11-09_12-02-07`

Need to install `peft` and `accelerate` to run the script:

```shell
pip install peft accelerate
```

## Evaluation

Use the provided script `fine-tuned-esmbind-inference-performance/r2/evaluate.py` to evaluate the performance of the fine-tuned model on the `AsEP` antigen.

This will create a file `esmbind_metrics.csv` to store the evaluation metrics.
