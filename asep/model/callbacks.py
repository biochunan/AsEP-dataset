"""
credit: https://stackoverflow.com/a/73704579
Example usage:
early_stopper = EarlyStopper(patience=3, min_delta=10)
for epoch in np.arange(n_epochs):
    train_loss = train_one_epoch(model, train_loader)
    validation_loss = validate_one_epoch(model, validation_loader)
    if early_stopper.early_stop(validation_loss):
        break
"""
import os
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Union
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import wandb

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s {%(pathname)s:%(lineno)d} [%(levelname)s] %(name)s - %(message)s [%(threadName)s]',
                    datefmt='%H:%M:%S')


class EarlyStopper:
    def __init__(
        self,
        patience=1,
        min_delta=0,
        minimize: bool = True,
        metric_name: str="val_loss"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.minimize = minimize
        self.best_val = np.inf if minimize else -np.inf
        self.metric_name = metric_name

    def early_stop(self, epoch: int, metrics: Dict):  # sourcery skip: merge-else-if-into-elif
        assert self.metric_name in metrics.keys(), f"provided metric_name {self.metric_name} not in metrics.\nValid keys are {metrics.keys()}"
        value = metrics[self.metric_name]
        # minimize
        if self.minimize:
            if value < self.best_val:
                self.reset_counter(value, epoch)
            elif value > (self.best_val + self.min_delta):
                self.counter += 1
                logging.info(f"Epoch {epoch}, EarlyStopper counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    return True
        else:
            if value > self.best_val:
                self.reset_counter(value, epoch)
            elif value <= (self.best_val - self.min_delta):
                self.counter += 1
                logging.info(f"Epoch {epoch}, EarlyStopper counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    return True
        return False

    def reset_counter(self, value: Tensor, epoch: int):
        """ reset counter and best_val

        Args:
            value (Tensor): metric value to be compared
            epoch (int): epoch number
        """
        self.best_val = value
        self.counter = 0
        logging.info(f"Epoch {epoch}, EarlyStopper reset counter")


class ModelCheckpoint:
    def __init__(
        self,
        save_dir: Union[Path, str],
        k: int = 1,
        minimize: bool = True,
        metric_name: str="val_loss"
    ):
        self.save_dir = Path(save_dir)
        self.k = k
        self.minimize = minimize
        self.metric_name = metric_name
        self.best_k_metric_value, self.best_k_epoch, self.best_k_fp = \
            [np.inf] * k, [-1] * k, [None] * k
        if not self.minimize:
            self.best_k_metric_value = [-np.inf] * k
        """
        best_k_metric_value: index pointing to the same epoch and file
        best_k_epoch       : index pointing to the same epoch and file
        best_k_fp          : index pointing to the same epoch and file
        """

        # create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # create interim directory to save state_dict files
        self.interim_dir = self.save_dir.joinpath("interim")
        self.interim_dir.mkdir(parents=True, exist_ok=True)

        # create best_k directory to save best k state_dict files
        self.best_k_dir = self.save_dir.joinpath(f"best_{k}")
        self.best_k_dir.mkdir(parents=True, exist_ok=True)

    def time_stamp(self):
        """ generate a time stamp
        e.g. 20230611-204118
        year month day - hour minute second
        2006 06    11  - 20   41     18

        Returns:
            _type_: str
        """
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def save_model(
            self,
            epoch: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            metric_value: Tensor
        ) -> Path:
        """ save a model to interim directory

        Args:
            epoch (int): epoch number
            model (nn.Module): model to save state_dict
            optimizer (torch.optim.Optimizer): optimizer to save state_dict
            metric_value (Tensor): metric value to save

        Returns:
            Path: path to the saved model
        """
        # prepare the object to save
        obj_to_save = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        # save the model to interim directory
        ckpt_path = self.interim_dir.joinpath(
                f"epoch{epoch}-{self.time_stamp()}.pt"
            )
        torch.save(obj_to_save, ckpt_path)
        return ckpt_path

    def update_best_k(self, epoch: int, metric_value: Tensor, ckpt_path: Path):
        """ Update the best k metric value, epoch number, and file path

        Args:
            epoch (int): epoch number
            metric_value (Tensor): metric value to compare with the best k metric value
            ckpt_path (Path): path to the saved model
        """
        # find the index -> worst model in the best k
        idx = self.best_k_metric_value.index(max(self.best_k_metric_value)) if self.minimize \
            else self.best_k_metric_value.index(min(self.best_k_metric_value))
        # update
        self.best_k_metric_value[idx] = metric_value
        self.best_k_epoch[idx]        = epoch
        self.best_k_fp[idx]           = ckpt_path

    def step(
        self,
        metrics: Dict,                    # dict of metrics
        epoch: int,                       # current epoch
        model: nn.Module,                 # model
        optimizer: torch.optim.Optimizer  # optimizer
        ):
        """
        Save a model if metric is better than the current best metric
        model is saved as a dictionary consisting of the following keys:
        - epoch: int
        - model_state_dict: (collections.OrderedDict)
            - keys: layer name + .weight e.g. odict_keys(['ab_hidden.0.weight', 'ag_hidden.0.weight', 'input_ab_layer.weight', 'input_ag_layer.weight'])
            - values: layer weights e.g. data["model_state_dict"]["input_ag_layer.weight"].shape => torch.Size([480, 128])
        - optimizer_state_dict: (dict)
            - keys: 'state', 'param_groups'
            - values: tensors
        - val_loss: Tensor

        Args:
            validation_loss (Tensor): validation loss from current epoch
        """
        assert self.metric_name in metrics.keys(), f"provided metric_name {self.metric_name} not in metrics.\nValid keys are {metrics.keys()}"
        v = metrics[self.metric_name]

        if (self.minimize and v < max(self.best_k_metric_value)) or \
            (not self.minimize and v > min(self.best_k_metric_value)):
                # save model
                ckpt_path = self.save_model(
                    epoch=epoch, model=model, optimizer=optimizer, metric_value=v)
                # update best k
                self.update_best_k(epoch=epoch, metric_value=v, ckpt_path=ckpt_path)

    def sort_best_k(self):
        """ sort the best k models return the indices
        the goal is to keep the best model at index 0
        - if minimize, the indices are in ascending order
        - if maximize, the indices are in descending order

        Returns:
            _type_: _description_
        """
        indices = torch.argsort(torch.stack(self.best_k_metric_value))  # indices in ascending order
        return indices if self.minimize else torch.flip(indices, dims=(0,))  # indices in descending order if not minimize

    def save_best_k(
        self,
        keep_interim: bool = True,
        ):
        """
        Save the best k models and the last model if save_last is True

        Args:
            keep_interim (bool): False to remove the interim directory
        """
        # sort best k
        indices = self.sort_best_k()  # the best at index 0
        # save the best k models to self.best_k_dir
        for i, j in enumerate(indices):
            # retrieve epoch and ckpt_path
            epoch, interim_ckpt_path = self.best_k_epoch[j], self.best_k_fp[j]
            """ option: create soft link to files in the interim directory
            # create a soft link in self.best_k_dir to that in the self.interim_dir
            # test if the soft link already exists
            if self.best_k_dir.joinpath(f"rank_{i}-epoch_{epoch}.pt").exists():
                # remove it
                os.remove(self.best_k_dir.joinpath(f"rank_{i}-epoch_{epoch}.pt"))
                # and issue a warning (this should not happen)
                logging.warn(f"soft link {self.best_k_dir.joinpath(f'rank_{i}-epoch_{epoch}.pt')} already exists. It is removed.")
            """
            # copy the file over to the self.best_k_dir
            shutil.copy(interim_ckpt_path, self.best_k_dir.joinpath(f"rank_{i}-epoch_{epoch}.pt"))

        # create a soft link to the best k models
        for i in range(self.k):
            dst = self.save_dir.joinpath(f"rank_{i}.pt")
            # if exist remove it
            if dst.exists():
                os.remove(dst)
                logging.warn(f"soft link {dst} already exists. It is removed.")
            os.symlink(src=os.path.relpath(list(self.best_k_dir.glob(f"rank_{i}*.pt"))[0], self.save_dir),
                       dst=dst)

        # remove the interim directory if keep_interim is False
        if not keep_interim:
            shutil.rmtree(self.interim_dir)

    def save_last(
        self,
        *args,
        upload: bool=True,
        wandb_run: wandb.sdk.wandb_run.Run=None,
        **kwargs
    ):
        """
        Wrapper to save the last model
        args and kwargs are passed to self.save_model
        Args:
            upload (bool): whether to upload the last model to wandb
                default is True
            wandb_run (wandb.sdk.wandb_run.Run): wandb run object
        """
        ckpt_path = self.save_model(*args, **kwargs)
        # copy self.interim_dir => self.save_dir
        shutil.copy(
            ckpt_path,
            self.save_dir
        )
        # if the soft link already exists, remove it
        if self.save_dir.joinpath("last.pt").exists():
            os.remove(self.save_dir.joinpath("last.pt"))
            logging.warn(f"soft link {self.save_dir.joinpath('last.pt')} already exists. It is removed.")
        # create a soft link to the last model
        os.symlink(
            src=os.path.relpath(self.save_dir.joinpath(ckpt_path.name), self.save_dir),
            dst=os.path.join(self.save_dir, "last.pt"),
        )

        if upload and wandb_run is not None:
            artifact = wandb.Artifact(
                name="last_epoch_checkpoint",
                type="model",
                metadata=dict(
                    metric_name=self.metric_name,
                ),
            )
            artifact.add_file(ckpt_path)
            wandb_run.log_artifact(artifact)

    def load_best(self) -> Dict[str, Any]:
        """
        Load the best model from the best_k_dir
        CAUTION: this should only be called when training is done
            i.e. after self.save_best_k() is called
        """
        return torch.load(
            self.save_dir.joinpath("rank_0.pt")
        )

    def upload_best_k_to_wandb(self, wandb_run: wandb.sdk.wandb_run.Run, suffix: str=None):
        """
        Upload the best k models to wandb as artifacts
        CAUTION: only call this after training is done, self.save_best_k() must be called
        NOTE: remove this dependency if needed in the future
        """
        suffix = suffix or ''
        # find the original path through the soft link
        artifact = wandb.Artifact(
            name="best_k_models" + suffix,
            type="model",
            metadata=dict(
                metric_name=self.metric_name,
            ),
        )
        for i in range(self.k):
            real_path = Path(os.path.realpath(self.save_dir.joinpath(f"rank_{i}.pt")))
            artifact.add_file(real_path)
        wandb_run.log_artifact(artifact)