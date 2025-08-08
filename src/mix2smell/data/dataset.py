import logging
import os
import pandas as pd
import torch
import math
import torch
from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from typing import Tuple, List, Dict, Union
from safetensors import safe_open
from safetensors.torch import save_file

from mix2smell.data.featurization import FeaturizeMolecules, FEATURIZATION_TYPE
from mix2smell.data.data import (
    MixtureDataInfo,
    COLUMN_PROPERTY,
    COLUMN_VALUE,
    COLUMN_TASKS,
)
from mix2smell.data.utils import pad_list, UNK_TOKEN



class Mix2SmellData(Dataset):
    """
    Base for handling a chemical mixture with associated properties.
    """

    def __init__(
        self,
        task: list[str],
        dataset: MixtureDataInfo,
        featurization: str = None,
    ):

        self.dataset = dataset
        self.inputs = dataset.inputs
        self.labels = dataset.labels

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Check if the provided task exists in the dataset
        for t in task:
            if t not in COLUMN_TASKS:
                raise ValueError(f"Task '{t}' is not in dataset: {COLUMN_TASKS}")
        
        self.task = task
        self.labels = self.labels[self.labels[task].any(axis=1)][["stimulus", self.dataset.metadata["columns"]["output_column"]]]
        self.inputs = self.inputs[self.inputs["stimulus"].isin(self.labels["stimulus"])]

        self.data = pd.merge(self.inputs, self.labels, on="stimulus")

        self.indices_tensor, self.fraction_tensor, self.output_tensor, self.feature_tensor = self.__tensorize__()

        self.featurization = featurization

        if self.featurization:
            self.features = self.__getfeatures__()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __num_unique_mixtures__(self) -> int:
        """
        Returns the number of unique mixtures in the dataset based on compound IDs.

        Returns:
            int: The number of unique mixtures.
        """
        cmp_ids = self.dataset.metadata["columns"]["id_column"]
        return len(self.data[cmp_ids].drop_duplicates())

    def __max_num_components__(self) -> int:
        """
        Returns the maximum number of components (compounds) in the dataset.

        Returns:
            int: Maximum number of components.
        """
        cmp_ids = self.dataset.metadata["columns"]["id_column"]

        def row_max_len(row):
            return max(len(row[col]) if isinstance(row[col], list) else 0 for col in cmp_ids)

        return self.data.apply(row_max_len, axis=1).max()

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tuple[int], Tuple[float], float]: A tuple consisting of:
                - Tuple of compound IDs
                - Tuple of mole fractions
                - Property value (label)
        """

        sample = {
            "ids": self.indices_tensor[idx],
            "fractions": self.fraction_tensor[idx],
            "label": self.output_tensor[idx],
        }

        if self.featurization:
            if FEATURIZATION_TYPE[self.featurization] == "graphs":
                sample["features"] = self.features # GraphBatch.from_data_list([feat[idx] for feat in self.features])
            elif FEATURIZATION_TYPE[self.featurization] == "tensors":
                sample["features"] = self.features[idx]

        return sample
    
    def __tensorize__(self) -> Dict:
        """
        Converts the dataset into tensor format for models.

        This method processes the input data and converts it into three main tensors:
        - indices_tensor: Contains the compound IDs with padding
        - fraction_tensor: Contains dilution/fraction information with padding
        - output_tensor: Contains the target labels

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - indices_tensor: Tensor of shape [N_POINTS, N_COMPONENTS, DIM_FEATURE, N_MIX] with compound IDs
                - fraction_tensor: Tensor of shape [N_POINTS, N_COMPONENTS, DIM_FEATURE, N_MIX] with dilution info
                - output_tensor: Tensor of shape [N_POINTS, N_TARGETS] with target values
                - (optional) feature_tensor: Tensor of shape [N_POINTS, N_FEATURES] with additional features originating from the dataset
        """
        id_col = self.dataset.metadata["columns"]["id_column"]
        fraction_col = self.dataset.metadata["columns"]["fraction_column"]
        output_col = self.dataset.metadata["columns"]["output_column"]
        feature_col = self.dataset.metadata["columns"]["feature_column"]
        max_length = self.__max_num_components__()

        # target tensor
        output_tensor = torch.tensor(self.data[output_col], dtype=torch.float)

        # this pads the mixtures up to `max_length`
        # `indices_tensor` gives final component id with padding value -999
        indices_tensors = []
        for col in id_col:
            col_tensor = torch.Tensor(self.data[col].apply(lambda x: pad_list(x, max_length=max_length)).tolist())
            indices_tensors.append(col_tensor)

        # indices_tensor = torch.stack(indices_tensors, dim=-1).unsqueeze(-1)
        indices_tensor = torch.stack(indices_tensors, dim=-1)

        # get fraction/dilution information if requested and available
        if len(fraction_col) == 0:
            print(f'No fraction/dilution information found, padding with {UNK_TOKEN}')
            fraction_tensor = torch.full_like(indices_tensor, fill_value=UNK_TOKEN).unsqueeze(-1)
        else:
            assert len(id_col) == len(fraction_col), 'Mismatch in fraction info and mixture rows'
            # same thing as indices_tensor, but with the dilution of components
            # padded components have padding value -999 as well
            fraction_tensors = []
            for i, col in enumerate(fraction_col):
                if col is not None:
                    if len(self.data[col]) > 1:
                        max_outer = len(self.data[col][0])
                    col_tensor = torch.tensor(self.data[col].apply(lambda x: pad_list(x, max_length=max_outer, max_inner_length=max_length)).tolist())
                else:
                    col_tensor = torch.full_like(indices_tensor[:, :, i], fill_value=UNK_TOKEN)
                fraction_tensors.append(col_tensor)

            # shape [N_POINTS, N_COMPONENTS, N_FRACTIONS, NUM_MIX]
            fraction_tensor = torch.stack(fraction_tensors, dim=-1)

            if fraction_tensor.dim() == 3:
                fraction_tensor = fraction_tensor.unsqueeze(-1)
            
            fraction_tensor = fraction_tensor.permute(0, 2, 1, 3) 

        if feature_col is None:
            return indices_tensor, fraction_tensor, output_tensor, None
        
        else:
            feature_tensors = []
            lofl = False
            for i, col in enumerate(feature_col):
                if type(self.data[col][0]) == str:
                    if self.data[col][0].count("{") == 0:
                        self.data[col] = self.data[col].apply(lambda x: [float(item) for item in x.replace("[","").replace("]","").replace(" ","").split(",")])
                    else: #list of lists, to do this, store the second inner list with different brackets ({})
                        lofl = True
                        self.data[col] = self.data[col].apply(lambda x: [[float(y.replace("{","").replace("}","").replace(" ","")) for y in item.split(",")] for item in x.replace("['{","").replace("}']","").replace(" ","").split("}','{")])
                if not lofl:
                    max_outer = len(self.data[col][0])
                    col_tensor = torch.tensor(self.data[col].apply(lambda x: pad_list(x, max_length=max_outer, max_inner_length=max_length)).tolist())
                    feature_tensors.append(col_tensor)
                else: # hard coded for the sake of time
                    col_tensor = torch.tensor(self.data[col].apply(lambda x: pad_list(x, max_length=10, max_inner_length=54)).tolist(), dtype=torch.float)
                    feature_tensors.append(col_tensor)
            return indices_tensor, fraction_tensor, output_tensor, feature_tensors


    def __getfeatures__(self) -> Dict:

        cache_dir = os.path.join(self.dataset.data_dir, f"{self.dataset.name.replace(' ', '_') + '_' + '_'.join(self.task)}_featurization")
        if FEATURIZATION_TYPE[self.featurization] == "graphs":
            cache_path = os.path.join(cache_dir, f"{self.featurization}.pt")
        elif FEATURIZATION_TYPE[self.featurization] == "tensors":
            cache_path = os.path.join(cache_dir, f"{self.featurization}.safetensors")

        # load from cache if found
        if os.path.exists(cache_path):
            if FEATURIZATION_TYPE[self.featurization] == "graphs":
                features = torch.load(cache_path, weights_only=False)
            elif FEATURIZATION_TYPE[self.featurization] == "tensors":
                with safe_open(cache_path, framework="pt") as f:
                    features = f.get_tensor(self.featurization)
        else:
            print("cache not found, featurizing dataset")
            featurizer = FeaturizeMolecules(self.featurization)
            compounds = self.dataset.compounds["SMILES"][self.dataset.compounds["SMILES"].notna()]

            if FEATURIZATION_TYPE[self.featurization] == "graphs":
                features = featurizer.featurize_graphs(
                    indices_tensor=self.indices_tensor, 
                    compounds=compounds,
                )
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(features, cache_path)
            elif FEATURIZATION_TYPE[self.featurization] == "tensors":
                if self.dataset.metadata["columns"]["feature_column"] is None:
                    features = featurizer.featurize_tensors(
                        indices_tensor=self.indices_tensor,
                        compounds=compounds,
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    save_file({self.featurization: features}, cache_path)
                else:
                    features = featurizer.featurize_tensors(
                        indices_tensor=self.indices_tensor,
                        compounds=compounds,
                        feature_tensor= self.feature_tensor
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    save_file({self.featurization: features}, cache_path)

        return features
