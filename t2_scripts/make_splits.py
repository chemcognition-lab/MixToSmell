import os, sys

from mix2smell.data.data import GSLFData, LBData, TrainingData
from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.splits import create_train_val_test_split, create_kfold_split

import torch
import numpy as np
from typing import Optional

seed = 0

#############
## Task1/2s/2m #
#############

data = TrainingData()
task = Mix2SmellData(dataset=data, task=["Task1", "Task2_single", "Task2_mix"], featurization="rdkit2d_normalized_features")

# binarize the labels (anything zero and non-zero)
labels = (torch.stack([t["label"] for t in task]) == 0).to(float)
create_train_val_test_split(
    task.dataset.name,
    task=task.task,
    mixture_indices_tensor=task.indices_tensor,
    target_label_tensor=labels,     # this stratifies it by label
    cache_dir=data.data_dir,
    test_size=0.2,
    seed=seed,
)

create_kfold_split(
    dataset_name= task.dataset.name,
    task=task.task,
    mixture_indices_tensor=task.indices_tensor,
    target_label_tensor=labels,      # this stratifies it by label
    cache_dir=data.data_dir,
    n_splits=5,
    seed=seed,
)
