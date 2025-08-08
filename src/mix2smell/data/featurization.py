from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from torch_geometric.data import Data, Batch
from typing import List, Optional, Dict
import logging
import torch
import numpy as np

from mordred import Calculator, descriptors
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from mix2smell.data.data import COLUMN_VALUE
from mix2smell.data.utils import indices_to_graphs, UNK_TOKEN

from .graph_utils import GraphBatch

from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

def parse_status(generator, smiles):
    results = generator.process(smiles)
    try: 
        processed, features = results[0], results[1:]
        if processed is None:
            logging.warning("Descriptastorus cannot process smiles %s", smiles)
        return features
    except TypeError:
        logging.warning("RDKit Error on smiles %s", smiles)
        # if processed is None, the features are are default values for the type


def morgan_fingerprints(
    smiles: List[str],
) -> torch.Tensor:
    """
    Builds molecular representation as a binary Morgan ECFP fingerprints with radius 3 and 2048 bits.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: tensor of shape [len(smiles), 2048] with ecfp featurized molecules

    """
    generator = MakeGenerator((f"Morgan3",))
    fps = torch.Tensor([parse_status(generator, x) for x in smiles])
    return fps


def rdkit2d_normalized_features(
    smiles: List[str],
) -> torch.Tensor:
    """
    Builds molecular representation as normalized 2D RDKit features.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: tensor of shape [len(smiles), 200] with featurized molecules

    """
    generator = MakeGenerator((f"rdkit2dhistogramnormalized",))
    fps = torch.Tensor([parse_status(generator, x) for x in smiles])
    return fps

def rdkit2d_normalized_features_with_t1(
    smiles: List[str],
) -> torch.Tensor:
    """
    Builds molecular representation as normalized 2D RDKit features.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: tensor of shape [len(smiles), 200] with featurized molecules

    """
    generator = MakeGenerator((f"rdkit2dhistogramnormalized",))
    fps = torch.Tensor([parse_status(generator, x) for x in smiles])
    return fps


def mordred_features(
    smiles: List[str],
) -> torch.Tensor:
    """
    Builds molecular representation as Mordred descriptors. 
    May be different sizes depending on set, and also may require scaling.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: tensor of shape [len(smiles), n_mordred] with featurized molecules

    """
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid]
    if len(valid_mols) == 0:
        features = torch.empty((0, 0))
    else:
        descs = calc.map(valid_mols)
        descs = np.array([list(d.values()) for d in descs], dtype=float)
        descs = descs[:, ~np.isnan(descs).any(axis=0)]      # some nans from mordred
        features = torch.tensor(np.array(descs), dtype=torch.float32)

    return features


def molt5_embedding(
    smiles: List[str],
):
    """
    Builds molecular representation using MolT5.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: tensor of shape [len(smiles), 1024]

    """
    model = PretrainedHFTransformer(kind='MolT5', notation='smiles', dtype=np.float32, device="cpu")
    print(model.device)
    embeds = torch.from_numpy(model(smiles))
    return embeds


def pyg_molecular_graphs(
    smiles: List[str], 
) -> List[Data]:
    """
    Convers a list of SMILES strings into PyGeometric molecular graphs.

    :param smiles: list of molecular SMILES
    :type smiles: list
    :return: list of PyGeometric molecular graphs
    """

    from torch_geometric.utils import from_smiles

    return [
        from_smiles(smiles=i) for i in smiles
    ]


def custom_molecular_graphs(
    smiles: List[str],
    init_globals: Optional[bool] = True,
) -> List[Data]:
    """
    Converts a list of SMILES strings into a custom graph tuple
    """

    from .graph_utils import from_smiles

    return [
        from_smiles(smiles=i, init_globals=init_globals) for i in smiles
    ]

FEATURIZATION_MAPPING = {
    "rdkit2d_normalized_features": rdkit2d_normalized_features,
    "morgan_fingerprints": morgan_fingerprints,
    "pyg_molecular_graphs": pyg_molecular_graphs,
    "molecular_graphs": custom_molecular_graphs,
    "molt5_embeddings": molt5_embedding,
    "mordred_features": mordred_features,
}

FEATURIZATION_TYPE = {
    "rdkit2d_normalized_features": "tensors",
    "morgan_fingerprints": "tensors",
    "pyg_molecular_graphs": "graphs",
    "molecular_graphs": "graphs",
    "molt5_embeddings": "tensors",
    "mordred_features": "tensors"
}


class FeaturizeMolecules(object):
    def __init__(self, featurization: str) -> None:

        if featurization not in FEATURIZATION_MAPPING:
            raise ValueError(f"Featurization type '{featurization}' is not recognized. Choose from {list(FEATURIZATION_MAPPING.keys())}.")

        self.featurization = featurization

    def featurize_tensors(self, indices_tensor: torch.Tensor, compounds: List[str], feature_tensor: torch.Tensor | None) -> torch.Tensor:

        featurized_mols = FEATURIZATION_MAPPING[self.featurization](compounds)

        all_features = []
        # indices_tensor specifies the compound index, padded up to NUM_COMPONENTS
        for ind_tensor in torch.unbind(indices_tensor, dim=-1):     # loop through N_MIXTURES
            featurized_tensor = torch.full(
                (ind_tensor.shape[0], ind_tensor.shape[1], featurized_mols.shape[1]),
                UNK_TOKEN,
                dtype=featurized_mols.dtype,
            )
            valid_indices = ind_tensor != UNK_TOKEN
            row_indices = ind_tensor.long()
            featurized_tensor[valid_indices] = featurized_mols[row_indices[valid_indices], :]

            all_features.append(featurized_tensor)
        
        # shape [N_POINTS, NUM_COMPONENTS, DIM_FEATURES, NUM_MIXTURES]
        # TODO also return a mask?

        if feature_tensor is None:
            featurized_tensor = torch.stack(all_features, dim=-1)

            return featurized_tensor
        
        else:
            if feature_tensor[-1].size()[1] == 1 or feature_tensor[-1].size()[1] == 53:
                return torch.cat((featurized_tensor, torch.stack(feature_tensor, dim = 1).repeat(1, 10, 1)), dim = 2).unsqueeze(-1)
            else:
                return torch.cat((featurized_tensor, feature_tensor[-1]), dim = 2).unsqueeze(-1)

    def featurize_graphs(self, indices_tensor: torch.Tensor, compounds: List[str]) -> List[Data]:

        featurized_mols = FEATURIZATION_MAPPING[self.featurization](compounds)
        
        # featurized_graphs = []
        # for ind_tensor in torch.unbind(indices_tensor, dim=-1):
        #     # have to do loop, lists of Data
        #     graph_components = []
        #     for r in ind_tensor.long():
        #         graph_components.append(
        #             GraphBatch.from_data_list([featurized_mols[i] for i in r if i != UNK_TOKEN])
        #         )
            
        #     # returns graphs batched per mixture
        #     featurized_graphs.append(GraphBatch.from_data_list(graph_components))
            
        return featurized_mols