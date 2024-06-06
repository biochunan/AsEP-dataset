import os
import pickle
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Set, Tuple, Union)

import numpy as np
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from datasets import Dataset
from loguru import logger
# Imports specific to the custom peft lora model
from peft import (LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_config,
                  get_peft_model)
from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)

# ==================== Configuration ====================
# Specify one GPU
if not os.environ["CUDA_VISIBLE_DEVICES"]:
    print('setting CUDA_VISIBLE_DEVICES=0')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# # overwrite if needed
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE='train'  # 'train' or 'debug'
BASE =Path("/home/chunan/UCL/scripts/FineTuningLargeModels/esmbind")
DATA = BASE/"binding_sites_random_split_by_family_550K"
INTERIM = BASE/"interim"

# ==================== Function ====================
# Helper Functions and Data Preparation
def truncate_labels(labels, max_length):
    """Truncate labels to the specified max_length."""
    return [label[:max_length] for label in labels]

def compute_metrics(p):
    """Compute metrics for evaluation."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove padding (-100 labels)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()

    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)

    # Compute precision, recall, F1 score, and AUC
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, predictions)

    # Compute MCC
    mcc = matthews_corrcoef(labels, predictions)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc}

def compute_loss(model, inputs):
    """Custom compute_loss function."""
    logits = model(**inputs).logits
    labels = inputs["labels"]
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    active_loss = inputs["attention_mask"].view(-1) == 1
    active_logits = logits.view(-1, model.config.num_labels)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

def _seconds_to_hms(seconds: float) -> str:
    """Convert seconds to h:m:s format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

# --------------------
# Define Custom Trainer Class
# --------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = compute_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss

# --------------------
# Training Function
# --------------------
def train_function_no_sweeps(train_dataset, test_dataset):
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

    # Define labels and model
    id2label = {0: "No binding site", 1: "Binding site"}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)

    # Convert the model into a PeftModel
    peft_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["query", "key", "value"], # also try "dense_h_to_4h" and "dense_4h_to_h"
        lora_dropout=config["lora_dropout"],
        bias="none", # or "all" or "lora_only"
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)

    # Use the accelerator
    model = accelerator.prepare(model)
    train_dataset = accelerator.prepare(train_dataset)
    test_dataset = accelerator.prepare(test_dataset)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Training setup
    training_args = TrainingArguments(
        output_dir=f"esm2_t12_35M-lora-binding-sites_{timestamp}",
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        gradient_accumulation_steps=1,
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=200,
        save_total_limit=7,
        no_cuda=False,
        seed=8893,
        fp16=True,
        report_to='wandb'
    )

    # Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train and Save Model
    trainer.train()
    accelerator.wait_for_everyone()
    logger.debug("Finished training.")

    logger.info("Saving model...")
    save_path = os.path.join("lora_binding_sites", f"best_model_esm2_t12_35M_lora_{timestamp}")
    logger.info(f"{save_path=}")

    # single GPU training: save the model using the Trainer's save method
    trainer.save_model(save_path)
    # # distributed training: save the model in the format of the original model
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(save_path)

    # Save the tokenizer
    tokenizer.save_pretrained(save_path)


# ==================== Main ====================
if __name__ == "__main__":
    # --------------------
    # Load the data
    # --------------------
    logger.info("Loading data...")

    # Load the data from pickle files (replace with your local paths)
    with open(DATA/"train_sequences_chunked_by_family.pkl", "rb") as f:
        train_sequences: List[str] = pickle.load(f)
        print(f"{len(train_sequences)=}")
        '''
        size: (N,)
        each element is a sequence of length L
        e.g. train_sequences[:2]
        ['MSTMMIFTGNANPDLAFKIANHLQVPLGQALVGKFSDGETM...'
            'MTHRFSTIQAAVDAMHRGEVIIVVDAEDRENEGDFVAAAEK...']
        '''

    with open(DATA/"test_sequences_chunked_by_family.pkl", "rb") as f:
        test_sequences: List[str] = pickle.load(f)
        print(f"{len(test_sequences)=}")
        '''
        size: (N, L)
        same format as train_sequences
        '''

    with open(DATA/"train_labels_chunked_by_family.pkl", "rb") as f:
        train_labels: List[List[int]] = pickle.load(f)
        print(f"{len(train_labels)=}")
        '''
        size: (N, L)
        each element is a list of binary int labels of length L
        if convert to string, e.g. ''.join([str(i) for i in train_labels[0]])
        000000000000000000000000000000000000111000000000000000000000000000000000000000...
        '''

    with open(DATA/"test_labels_chunked_by_family.pkl", "rb") as f:
        test_labels = pickle.load(f)
        print(f"{len(test_labels)=}")

    logger.info("Data loaded.")

    if MODE == 'debug':
        # take only the first k sequences for debugging
        k = 1000
        train_sequences = train_sequences[:k]
        test_sequences = test_sequences[:k]
        train_labels = train_labels[:k]
        test_labels = test_labels[:k]

    # --------------------
    # Tokenization
    # --------------------
    logger.info("Loading tokenizer...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    max_sequence_length = 1000

    end_time = time.time()
    elapsed_time = _seconds_to_hms(end_time - start_time)
    logger.info(f"Loading tokenizer, time elapsed: {elapsed_time}")  # super fast

    # Tokenize sequences
    def _get_tokenized_seqs(set_name: str, seqs: List[str], mode: str) -> Dict[str, torch.Tensor]:
        if mode == 'train':
            fp: Path = INTERIM/f"{set_name}_tokenized.pt"
        elif mode == 'debug':
            fp: Path = INTERIM/f"{set_name}_tokenized_debug.pt"
        else:
            raise ValueError("mode must be either 'train' or 'debug'")

        if fp.exists():
            tokenized = torch.load(fp)
        else:
            tokenized = tokenizer(seqs, padding=True, truncation=True,
                                    max_length=max_sequence_length,
                                    return_tensors="pt",
                                    is_split_into_words=False)
            torch.save(tokenized, fp)

        return tokenized

    logger.info("Tokenizing training sequences...")
    start_time = time.time()

    train_tokenized = _get_tokenized_seqs('train', train_sequences, mode=MODE)

    ''' Annotations
    train_tokenized.keys()                   # dict_keys(['input_ids', 'attention_mask'])
    train_tokenized['input_ids'].shape       # torch.Size([9876, 1000])
    train_tokenized['attention_mask'].shape  # torch.Size([9876, 1000])

    x = train_sequences[0]
    y = train_tokenized['input_ids'][0]
    z = train_tokenized['attention_mask'][0]

    len(train_sequences[0])  # 315
    s =  '<cls> ' + \
         ' '.join(map(lambda i: str(f'{i:<2}'), x[:9])) + ' ... ' + \
         ' '.join(map(lambda i: str(f'{i:<2}'), x[312:])) + \
         '<eos>' + \
         ' '.join(['<pab>']*3) + ' ... \n'
    s += f'{y[0]:<5} ' + \
         ' '.join(map(lambda i: str(f'{i:<2}'), y[1:10])) + ' ... ' + \
         ' '.join(map(lambda i: str(f'{i:<2}'), y[313:316])) + \
         f'{y[316]:<5} ' + \
         ' '.join(map(lambda i: str(f'{i:<5}'), y[317:320])) + '...' + '\n'
    s += f'{z[0]:<5} ' + \
         ' '.join(map(lambda i: str(f'{i:<2}'), z[1:10])) + ' ... ' + \
         ' '.join(map(lambda i: str(f'{i:<2}'), z[313:316])) + \
         f'{z[316]:<5} ' + \
         ' '.join(map(lambda i: str(f'{i:<5}'), z[317:320])) + '...' + '\n'
    print(s)
    tokenizer.all_tokens[18]

    len(x)  # 315
    torch.where(y == 2)  # 316
    torch.sum(y == 1)  # 683, count paddings
    torch.sum(z == 1)  # 317, all non-padding tokens are included in attention_mask
    '''

    end_time = time.time()
    logger.info(f"Tokenizing training sequences, time elapsed: {_seconds_to_hms(end_time - start_time)}")
    logger.info("Tokenizing training sequences... Done")  # 6 min 20 seconds

    # tokenizing test seqeunces
    logger.info("Tokenizing test sequences...")
    start_time = time.time()

    test_tokenized = _get_tokenized_seqs('test', test_sequences, mode=MODE)

    end_time = time.time()
    logger.info(f"Tokenizing test sequences, time elapsed: {_seconds_to_hms(end_time - start_time)}")
    logger.info("Tokenizing test sequences... Done")  # 1 min 29 seconds

    # --------------------
    # Directly truncate the
    # entire list of labels
    # --------------------
    logger.info("Truncating labels...")
    start_time = time.time()

    train_labels = truncate_labels(train_labels, max_sequence_length)
    test_labels = truncate_labels(test_labels, max_sequence_length)

    end_time = time.time()
    logger.info(f"Truncating labels, time elapsed: {_seconds_to_hms(end_time - start_time)}")
    logger.info("Truncating labels... Done")

    # --------------------
    # Create datasets
    # --------------------
    logger.info("Creating datasets...")

    def _get_datasets(set_name: str, tokenized: Dict[str, torch.Tensor], labels: List[List[int]], mode: str) -> Dataset:
        if mode == 'train':
            fp: Path = INTERIM/f"{set_name}_dataset"
        elif mode == 'debug':
            fp: Path = INTERIM/f"{set_name}_dataset_debug"
        else:
            raise ValueError("mode must be either 'train' or 'debug'")

        if fp.exists():
            dataset = Dataset.load_from_disk(fp)
        else:
            dataset = Dataset.from_dict({k: v for k, v in tokenized.items()}).add_column("labels", labels)
            dataset.save_to_disk(fp)

        return dataset

    train_dataset = _get_datasets('train', train_tokenized, train_labels, mode=MODE)
    test_dataset = _get_datasets('test', test_tokenized, test_labels, mode=MODE)

    '''Annotations
    test_dataset[0].keys()             # dict_keys(['input_ids', 'attention_mask', 'labels'])
    test_dataset[0]['input_ids']       # List[int]      length 1000
    test_dataset[0]['attention_mask']  # List[int] 0/1  length 1000
    test_dataset[0]['labels']          # List[int] 0/1  length 315
    '''

    logger.info("Creating datasets... Done")  # takes 6 min

    # --------------------
    # Compute Class Weights
    # --------------------
    logger.info("Computing class weights...")

    classes = [0, 1]
    flat_train_labels = [label for sublist in train_labels for label in sublist]
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)
    accelerator = Accelerator()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(accelerator.device)

    logger.info("Computing class weights... Done")  # takes 20 seconds

    '''Annotations
    len(train_labels)       # 450,330
    len(flat_train_labels)  # 208,280,718
    class_weights           # tensor([ 0.5157, 16.4235])

    # this is not calculated by the fraction of each type of class labels
    flat_train_labels.count(1) / len(flat_train_labels)  # 0.030444162382808764
    flat_train_labels.count(0) / len(flat_train_labels)  # 0.9695558376171912

    # Instead, compute_class_weight uses the following formula
    #     n_samples / (n_classes * np.bincount(y))
    # Here n_samples is the total number of labels i.e. total number of tokens
    np.bincount(flat_train_labels)  # => array([201,939,786,   6,340,932]) count of 0 and 1 labels
    208280718 / (2 * 201939786)     # => 0.5157000562534022
    208280718 / (2 * 6340932)       # => 16.423509824738698
    '''

    # --------------------
    # Train the model
    # --------------------
    logger.info("Training the model...")
    start_time = time.time()

    train_function_no_sweeps(train_dataset, test_dataset)

    end_time = time.time()
    logger.info(f"Training the model, time elapsed: {_seconds_to_hms(end_time - start_time)}")
    logger.info("Training the model... Done")  # 18 hours on a single GPU
