import sys
import functools
from omegaconf import OmegaConf
from mix2smell.model.aggregation import (
    MeanAggregation,
    MaxAggregation,
    PrincipalNeighborhoodAggregation,
    PNAMixtureSizeScaled,
    AttentionAggregation,
    Set2SetAggregation,
)

from mix2smell.model.predictor import (
    PredictiveHead,
    PhysicsPredictiveHead,
    ScaledCosineRegressor,
    HurdleHead,
    AdaptiveOutputLayer,
    TwoHeadRegressor,
)
from mix2smell.model.utils import ActivationEnum
from mix2smell.model.graph import GraphNets
from mix2smell.model.linear import FullyConnectedNet
from mix2smell.model.mixture import SelfAttentionBlock, DeepSet, MixtureModel
from mix2smell.model.pom import POMModel
from mix2smell.data.graph_utils import NODE_DIM, EDGE_DIM


import torch
from torch import nn


if sys.version_info >= (3, 11):
    import enum
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


# Molecule block
class MoleculeEncoderEnum(StrEnum):
    """Basic str enum for molecule encoders (en-to-end training)."""

    gnn = enum.auto()
    linear = enum.auto()

# Molecular Fractions
class FractionsAggregationEnum(StrEnum):
    """Basic str enum for mole fraction aggregation (en-to-end training)."""
    concat = enum.auto()
    multiply = enum.auto()


# Mixture block
class MixtureEncoderEnum(StrEnum):
    """Basic str enum for mixture encoders (en-to-end training)."""

    deepset = enum.auto()
    self_attn = enum.auto()

# Mixture block
class AggEnum(StrEnum):
    """Basic str enum for molecule aggregators."""

    mean = enum.auto()
    max = enum.auto()
    pna = enum.auto()
    scaled_pna = enum.auto()
    attn = enum.auto()
    set2set = enum.auto()

# Prediction head
class RegressorEnum(StrEnum):
    """Basic str enum for regressors."""
    mlp = enum.auto()
    hurdle = enum.auto()
    twohead = enum.auto()



def build_pom_model(config):

    # Molecule block
    mol_encoder_methods = {
        MoleculeEncoderEnum.gnn: GraphNets(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=config.mol_encoder.gnn.global_dim,
            hidden_dim=config.mol_encoder.gnn.hidden_dim,
            depth=config.mol_encoder.gnn.depth,
        )
    }

    # Prediction head
    regressor_type = {
        RegressorEnum.mlp: PredictiveHead(
            hidden_dim=config.regressor.hidden_dim,
            num_layers=config.regressor.num_layers,
            output_dim=config.regressor.mlp.output_dim,
            dropout_rate=config.dropout_rate,
        ),
        RegressorEnum.hurdle: HurdleHead(
            hidden_dim=config.regressor.hidden_dim,
            num_layers=config.regressor.num_layers,
            output_dim=config.regressor.mlp.output_dim,
            dropout_rate=config.dropout_rate,
        ),
        RegressorEnum.twohead: TwoHeadRegressor(
            hidden_dim=config.regressor.hidden_dim,
            num_layers=config.regressor.num_layers,
            output_dim=config.regressor.mlp.output_dim,
            dropout_rate=config.dropout_rate,
        ),
    }

    pom_model = POMModel(
        mol_encoder=mol_encoder_methods[config.mol_encoder.type],
        regressor=regressor_type[config.regressor.type],
        fraction_aggregation_type=config.fraction_aggregation.type,
        fraction_film_activation=config.fraction_aggregation.film.activation,
        fraction_film_output_dim=config.fraction_aggregation.film.output_dim,
    )

    # load the various pretrained models if specified
    pretrained_path = config.mol_encoder.get('pretrained_path', None)
    if pretrained_path is not None:
        print('Pretrained path found, and loaded for mol_encoder.')
        state_dict = torch.load(pretrained_path)
        pom_model.mol_encoder.load_state_dict(state_dict)

    pretrained_path = config.regressor.get('pretrained_path', None)
    if pretrained_path is not None:
        print('Pretrained path found, and loaded for regressor.')
        state_dict = torch.load(pretrained_path)
        pom_model.regressor.load_state_dict(state_dict) 

    return pom_model


def build_mixture_model(config):

    # Molecule block
    mol_encoder_methods = {
        MoleculeEncoderEnum.gnn: GraphNets(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=config.mol_encoder.gnn.global_dim,
            hidden_dim=config.mol_encoder.gnn.hidden_dim,
            depth=config.mol_encoder.gnn.depth,
        ),
        MoleculeEncoderEnum.linear: nn.LazyLinear(
            # config.mol_encoder.gnn.global_dim,
            config.mol_encoder.output_dim,
        ),
    }

    # Projection layer
    project_input = nn.LazyLinear(config.attn_aggregation.embed_dim)

    # Mixture block
    mol_aggregation_methods = {
        AggEnum.mean: MeanAggregation(),
        AggEnum.max: MaxAggregation(),
        AggEnum.pna: PrincipalNeighborhoodAggregation(),
        AggEnum.scaled_pna: PNAMixtureSizeScaled(),
        AggEnum.attn: AttentionAggregation(
            embed_dim=config.attn_aggregation.embed_dim,
            num_heads=config.attn_num_heads,
            dropout_rate=config.dropout_rate,
        ),
        AggEnum.set2set: Set2SetAggregation(
            in_channels=config.set2set_aggregation.in_channels,
            processing_steps=config.set2set_aggregation.processing_steps,
        )
    }

    # Mixture block
    mixture_encoder_methods = {
        MixtureEncoderEnum.self_attn: SelfAttentionBlock(  # TODO: Rename
            num_layers=config.mix_encoder.num_layers,
            embed_dim=config.mix_encoder.embed_dim,
            num_heads=config.attn_num_heads,
            add_mlp=config.mix_encoder.self_attn.add_mlp,
            dropout_rate=config.dropout_rate,
            mol_aggregation=mol_aggregation_methods[config.mol_aggregation],
        ),
        MixtureEncoderEnum.deepset: DeepSet(
            embed_dim=config.mix_encoder.embed_dim,
            num_layers=config.mix_encoder.num_layers,
            dropout_rate=config.dropout_rate,
            mol_aggregation=mol_aggregation_methods[config.mol_aggregation],
        ),
    }

    # Prediction head
    regressor_type = {
        RegressorEnum.mlp: PredictiveHead(
            hidden_dim=config.regressor.hidden_dim,
            num_layers=config.regressor.num_layers,
            output_dim=config.regressor.mlp.output_dim,
            dropout_rate=config.dropout_rate,
        ),
    }

    mixture_model = MixtureModel(
        mol_encoder=mol_encoder_methods[config.mol_encoder.type],
        projection_layer=project_input,
        mix_encoder=mixture_encoder_methods[config.mix_encoder.type],
        regressor=regressor_type[config.regressor.type],
        fraction_aggregation_type=config.fraction_aggregation.type,
        fraction_film_activation=config.fraction_aggregation.film.activation,
        fraction_film_output_dim=config.fraction_aggregation.film.output_dim,
    )

    return mixture_model

def build_adaptive_pom_model(
    config,
    new_output_dim: int,
    regressor_checkpoint_path: str,
    encoder_checkpoint_path: str,
    freeze_old: bool = True,
):
    """
    Builds a POMModel with an adaptive output layer for transfer learning.

    Args:
        config: OmegaConf config for model construction.
        new_output_dim: Output dimension for the new dataset.
        regressor_checkpoint_path: Path to the pretrained regressor checkpoint.
        encoder_checkpoint_path: Path to the pretrained encoder checkpoint.
        freeze_old: Whether to freeze the pretrained head weights.

    Returns:
        POMModel with AdaptiveOutputLayer as regressor.
    """
    # 1) Build molecule encoder
    mol_encoder = GraphNets(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        global_dim=config.mol_encoder.gnn.global_dim,
        hidden_dim=config.mol_encoder.gnn.hidden_dim,
        depth=config.mol_encoder.gnn.depth,
    )
    encoder_output_dim = config.mol_encoder.output_dim

    # 2) Load pretrained regressor checkpoint
    reg_ckpt = torch.load(regressor_checkpoint_path, map_location='cpu')
    reg_state = reg_ckpt.get('state_dict', reg_ckpt)

    # 3) Infer pretrained head dims
    if 'layers.weight' not in reg_state:
        raise ValueError("Could not find 'layers.weight' in regressor checkpoint")
    weight = reg_state['layers.weight']  # shape: [old_output_dim, old_input_dim]
    old_output_dim, old_input_dim = weight.shape[0], weight.shape[1]

    # 4) Create adaptive regressor
    adaptive_regressor = AdaptiveOutputLayer(
        encoder_output_dim=encoder_output_dim,
        old_head_input_dim=old_input_dim,
        old_output_dim=old_output_dim,
        new_output_dim=new_output_dim,
        freeze_old=freeze_old,
    )
    # Initialize layers via dummy forward
    with torch.no_grad():
        _ = adaptive_regressor(torch.zeros(1, encoder_output_dim))

    # 5) Assemble POMModel
    pom_model = POMModel(
        mol_encoder=mol_encoder,
        regressor=adaptive_regressor,
        fraction_aggregation_type=config.fraction_aggregation.type,
        fraction_film_activation=config.fraction_aggregation.film.activation,
        fraction_film_output_dim=config.fraction_aggregation.film.output_dim,
    )

    # 6) Load encoder weights
    enc_state = torch.load(encoder_checkpoint_path, map_location='cpu')
    pom_model.mol_encoder.load_state_dict(enc_state)

    # 7) Load pretrained head weights into old_head
    remapped = { 'weight': reg_state['layers.weight'], 'bias': reg_state['layers.bias'] }
    pom_model.regressor.old_head.load_state_dict(remapped)

    return pom_model
