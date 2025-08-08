from pathlib import Path
import sys
import os
import numpy as np

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
sys.path.append(str(src_path))

from mix2smell.data.data import TrainingData, LBData, KellerData
from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.splits import SplitLoader

# Call the classes found in src/mix2smell/data/data.py 
# For the dream challenge, we will have 3 classes, for the training, lb, and test data
# Within each, the data from Task1 and Task2 is merged so we can train on both if needed
training_data = TrainingData()
leaderboard_data = LBData()
keller_data = KellerData()

# Call the torch dataset wrapper found in src/mix2smell/data/dataset.py
training_task = Mix2SmellData(
    dataset=keller_data,
    task=["keller"],  # Specify which Task of the challenge you want to load (here, we load them all)
    featurization="rdkit2d_normalized_features",  # Specify which featurization you want to use, see FEATURIZATION_TYPE variable in src/mix2smell/data/featurization.py for more info
)

# Get data stats
print(training_task.__len__())
print(training_task.__max_num_components__())
print(training_task.__num_unique_mixtures__())

# Get one data point
data_point = training_task.__getitem__(0)
print("All tensors in a datapoint:", data_point.keys())

# Contains the padded custom ID (NOT the ones used in the Dream original files)
print("ids tensor shape:", data_point["ids"].shape)

# Contains the padded dilution factor + the one-hot encoded solvent type
print("fractions tensor shape:", data_point["fractions"].shape)

# Contains the 51-dimensional label vector
print("label tensor shape:", data_point["label"].shape)

# Contains the padded molecular feature vectors, based on selected featurization type
print("features shape:", data_point["features"].shape)

# Load a split, previously made using the make_splits.py scripts. For now we only have random kfold split.
split_loader = SplitLoader(split_type="kfold")

train_indices, val_indices, test_indices = split_loader(
            dataset_name= training_task.dataset.name,
            task=training_task.task,
            cache_dir=training_task.dataset.data_dir,
            split_num=0,
        )
