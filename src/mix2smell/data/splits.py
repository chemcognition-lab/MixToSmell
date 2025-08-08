import os
import sys
import torch
import pandas as pd
import numpy as np
import glob

from typing import Tuple, Optional, List
from sklearn.model_selection import KFold, train_test_split
from mix2smell.data.utils import UNK_TOKEN
from mix2smell.model.utils import set_seed
from torch.utils.data import random_split

from safetensors import safe_open
from safetensors.torch import save_file
from collections import Counter

from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification


def calculate_inner_lengths(row):
    return [len(inner_arr) for inner_arr in row]


def split_indices(indices, valid_percent, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    shuffled = indices[torch.randperm(len(indices))]
    train_size = int((1 - valid_percent) * len(indices))
    return shuffled[:train_size], shuffled[train_size:]


def create_kfold_split(
    dataset_name: str,
    task: list[str],
    mixture_indices_tensor: torch.Tensor,
    cache_dir: str,
    target_label_tensor: torch.Tensor = None,
    n_splits: Optional[int] = 5,
    val_size: Optional[float] = 0.1,
    seed: Optional[int] = None,
) -> None:
    """
    Creates k-fold splits for a dataset and saves the train/val/test indices for each fold.

    Args:
        dataset_name (str): Name of the dataset.
        task (list[str]): List of task names.
        mixture_indices_tensor (torch.Tensor): Tensor containing the indices of the mixtures.
        cache_dir (str): Directory to cache the split indices.
        target_label_tensor (torch.Tensor, optional): Tensor of target labels for stratification. If None, no stratification is used.
        n_splits (int, optional): Number of folds for k-fold splitting. Default is 5.
        val_size (float, optional): Fraction of the data to use for validation within the train/val split. Default is 0.1.
        seed (int, optional): Random seed for reproducibility.
    """

    save_path = os.path.join(cache_dir, f"{dataset_name.replace(' ', '_')}_{'_'.join(task)}_splits")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    set_seed(seed)
    dataset_indices_tensor = torch.arange(mixture_indices_tensor.shape[0])

    train_val_frac = 1.0 - 1.0 / n_splits  # gives 20 for test set
    train_frac = (train_val_frac - val_size)/ train_val_frac  # get 70/10 of total for train/val sets
    
    if target_label_tensor is not None:
        # use iterative stratification
        target_label_tensor = target_label_tensor.numpy()
        kf = IterativeStratification(n_splits=n_splits, order=1)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for i, (train_indices, test_indices) in enumerate(
        kf.split(np.arange(mixture_indices_tensor.shape[0]).reshape(-1,1), target_label_tensor)
    ):
        # do a split for 70/10/20 train/validation/test
        if target_label_tensor is not None:
            # stratify
            train_indices, _, val_indices, _ = iterative_train_test_split(
                train_indices.reshape(-1, 1), target_label_tensor[train_indices],
                test_size=1.0-train_frac
            )
            train_indices = train_indices.flatten()
            val_indices = val_indices.flatten()
        else:
            train_indices, val_indices = train_test_split(
                train_indices, train_size=train_frac, random_state=seed
            )

        train_indices = dataset_indices_tensor[train_indices]
        val_indices = dataset_indices_tensor[val_indices]
        test_indices = dataset_indices_tensor[test_indices]

        cache_path = os.path.join(save_path, f"kfold_split_{i}.safetensors")

        save_file(
            {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices},
            cache_path,
        )


def create_train_val_test_split(
    dataset_name: str,
    task: list[str],
    mixture_indices_tensor: torch.Tensor,
    cache_dir: str,
    target_label_tensor: torch.Tensor = None,
    test_size: Optional[float] = 0.2,
    val_size: Optional[float] = 0.1,
    seed: Optional[int] = 0,
):
    """
    Splits a dataset into train, validation, and test sets, with optional stratification.

    Args:
        dataset_name (str): Name of the dataset.
        task (list[str]): List of task names.
        mixture_indices_tensor (torch.Tensor): Tensor of mixture/sample indices.
        cache_dir (str): Directory to save the split indices.
        target_label_tensor (torch.Tensor, optional): Tensor of target labels for stratification. If None, random split is used.
        test_size (float, optional): Fraction of the dataset to use as the test set. Default is 0.2.
        val_size (float, optional): Fraction of the dataset to use as the validation set. Default is 0.1.
        seed (int, optional): Random seed for reproducibility. Default is 0.
    """

    save_path = os.path.join(cache_dir, f"{dataset_name.replace(' ', '_')}_{'_'.join(task)}_splits")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    set_seed(seed)
    dataset_indices_tensor = torch.arange(mixture_indices_tensor.shape[0])

    # Ensure test_size and val_size do not overlap and sum to less than 1
    assert (test_size + val_size) < 1.0, "test_size + val_size must be less than 1.0"
    val_frac = val_size / (1.0 - test_size)

    if target_label_tensor is not None:
        num_dat = len(target_label_tensor)
        # split training + val
        train_val_ind, _, test_ind, _ = iterative_train_test_split(
            np.array(range(num_dat)).reshape(-1, 1), target_label_tensor.numpy(), 
            test_size=test_size
        )

        # split val
        train_ind, _, val_ind, _ = iterative_train_test_split(
            train_val_ind, target_label_tensor.numpy()[train_val_ind.flatten()], 
            test_size=val_frac
        )

        train_ind = train_ind.flatten()
        val_ind = val_ind.flatten()
        test_ind = test_ind.flatten()

    else:
        train_val_ind, test_ind = train_test_split(
            dataset_indices_tensor, test_size=test_size, random_state=seed
        )

        train_ind, val_ind = train_test_split(
            train_val_ind, test_size=val_frac, random_state=seed
        )

    # get the indices
    train_indices = dataset_indices_tensor[train_ind]
    val_indices = dataset_indices_tensor[val_ind]
    test_indices = dataset_indices_tensor[test_ind]

    cache_path = os.path.join(save_path, f"tvt_split_0.safetensors")

    save_file(
        {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices},
        cache_path,
    )


SPLIT_MAPPING = {
    "kfold": create_kfold_split,
    "tvt": create_train_val_test_split,
}


class SplitLoader(object):

    def __init__(self, 
        dataset_name: str,
        task: list[str],
        cache_dir: str,
        split_type: str = "kfold"
    ) -> None:

        self.save_path = os.path.join(cache_dir, f"{dataset_name.replace(' ', '_') + '_' + '_'.join(task)}_splits")

        if split_type not in SPLIT_MAPPING:
            raise ValueError(f"Split type '{split_type}' is not recognized. Choose from {list(SPLIT_MAPPING.keys())}.")

        self.split_type = split_type
    
    def __len__(self):
        return len(glob.glob(f'{self.save_path}/{self.split_type}*'))

    def __call__(
        self,
        split_num: Optional[int] = 0,
    ):

        cache_path = os.path.join(self.save_path, f"{self.split_type}_split_{split_num}.safetensors")
        if os.path.exists(cache_path):
            with safe_open(cache_path, framework="pt") as f:
                train_indices = f.get_tensor("train_indices")
                val_indices = f.get_tensor("val_indices")
                test_indices = f.get_tensor("test_indices")
        else:
            print(f'Splits not found in {cache_path}')
            return

        return train_indices, val_indices, test_indices