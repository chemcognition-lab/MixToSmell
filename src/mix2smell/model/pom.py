import sys
import torch
from torch import nn
import torch.nn.functional as F

from mix2smell.data.utils import UNK_TOKEN
from mix2smell.model.utils import compute_key_padding_mask, ACTIVATION_MAP
from mix2smell.model.tensor_types import (
    Tensor,
    MixTensor,
    MaskTensor,
    EmbTensor,
    ManyEmbTensor,
    ManyMixTensor,
    PredictionTensor,
)

from typing import Union, Optional
import torch_geometric as pyg
from torch_geometric.data import Batch
from torch_geometric.nn.conv import FiLMConv
from mix2smell.model.predictor import PhysicsPredictiveHead, ScaledCosineRegressor


class MLP(nn.Module):
    """Basic MLP with dropout and GELU activation."""

    def __init__(
        self,
        hidden_dim: int,
        add_linear_last: bool,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(nn.LazyLinear(hidden_dim))
            self.layers.append(nn.GELU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(p=dropout_rate))
        if add_linear_last:
            self.layers.append(nn.LazyLinear(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        output = self.layers(x)
        return output


class FiLMLayer(nn.Module):
    def __init__(
        self,
        output_dim: int,
        act: str,
    ):
        super().__init__()

        self.gamma = nn.LazyLinear(output_dim)
        self.act = ACTIVATION_MAP[act]()
        self.beta = nn.LazyLinear(output_dim)

    def forward(self, x, condition):
        gamma = self.act(self.gamma(condition))
        beta = self.act(self.beta(condition))

        return gamma * x + beta


class AddNorm(nn.Module):
    """Residual connection with layer normalization and dropout."""

    def __init__(self, embed_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.norm(x1 + self.dropout(x2))


class POMModel(nn.Module):
    def __init__(
        self,
        mol_encoder: nn.Module,
        regressor: nn.Module,
        fraction_aggregation_type: str,
        fraction_film_activation: Optional[str] = None,
        fraction_film_output_dim: Optional[int] = None,
    ):
        super(POMModel, self).__init__()
        self.mol_encoder = mol_encoder
        self.regressor = regressor
        self.unk_token = UNK_TOKEN
        self._proj_frac = nn.Linear(198, 196)
        self.fraction_aggregation_type = fraction_aggregation_type

        if self.fraction_aggregation_type == "film":
            self.fraction_film = FiLMLayer(
                output_dim=fraction_film_output_dim,
                act=fraction_film_activation,
            )

    def add_fraction_information(
        self,
        mol_emb: torch.Tensor,
        x_fractions: torch.Tensor,
        fraction_aggregation_type: str,
    ) -> torch.Tensor:

        # Do not add dummy context
        if torch.all(x_fractions == self.unk_token):
            mol_emb = mol_emb
        else:
            # "select out" any mixture dimensions
            x_fractions = x_fractions[:, :, :, 0]      # "select" the first component, and the first mixture
            if fraction_aggregation_type == "concat":
                mol_emb = torch.concat((mol_emb, x_fractions), dim=-1)
            elif fraction_aggregation_type == "multiply":
                x_fractions = torch.where(x_fractions == self.unk_token, torch.tensor(1.0), x_fractions)
                mol_emb = mol_emb * x_fractions
            elif fraction_aggregation_type == "film":
                padding_mask = compute_key_padding_mask(mol_emb, self.unk_token)
                mol_emb = self.fraction_film(x=mol_emb, condition=x_fractions)
                pad_fill = torch.full_like(mol_emb, self.unk_token)
                mol_emb = torch.where(~padding_mask.unsqueeze(-1) , mol_emb, pad_fill)

        return mol_emb

    def embed(
        self, 
        x: torch.Tensor | pyg.data.Batch,
        x_ids: torch.Tensor,
    ):
        # we are working with single molecules that are going to be presented as mixtures
        # so we will select out the dimensions that don't matter
        embedding = self.mol_encoder(x, x_ids[:,:,0])   # selecting the "first mixture"
        return embedding

    def embed_with_fractions(
        self,
        x: torch.Tensor | pyg.data.Batch,
        x_ids: torch.Tensor,
        x_fractions: torch.Tensor = None,
    ):
        embedding = self.embed(x, x_ids)
        embedding_with_fractions = self.add_fraction_information(embedding, x_fractions, self.fraction_aggregation_type)
        return embedding_with_fractions

    def forward(
        self,
        x: torch.Tensor | pyg.data.Batch,
        x_ids: torch.Tensor,
        x_fractions: torch.Tensor,
    ) -> PredictionTensor:
        
        # --- DEBUG: show the raw inputs coming in ---
        # print(f"[DEBUG] POMModel.forward ▶ x_ids.shape:      {tuple(x_ids.shape)}")
        # print(f"[DEBUG] POMModel.forward ▶ x_fractions.shape: {tuple(x_fractions.shape)}")

        mol_emb = self.embed_with_fractions(
            x=x, x_ids=x_ids, x_fractions=x_fractions
        )
        # how many dims and what is feature-size right after embedding?
        # print(f"[DEBUG] POMModel.forward ▶ after embed_with_fractions: {tuple(mol_emb.shape)}")


        mol_emb = mol_emb[:,0,:]        # select the "first" component
        # print(f"[DEBUG] POMModel.forward ▶ after selecting first component: {tuple(mol_emb.shape)}")

        if mol_emb.size(-1) == 198:
            # if we have 198 features, we need to project them down to 196
            mol_emb = self._proj_frac(mol_emb)
            # print(f"[DEBUG] POMModel.forward ▶ after projection: {tuple(mol_emb.shape)}")
        pred = self.regressor(mol_emb)
        return pred

    def forward_mix_hurdle(
        self,
        x: torch.Tensor | pyg.data.Batch,
        x_ids: torch.Tensor,
        x_fractions: torch.Tensor,
    ) -> PredictionTensor:

        mol_emb = self.embed_with_fractions(
            x=x, x_ids=x_ids, x_fractions=x_fractions
        )

        clas = torch.full(
            (x_ids.shape[0], x_ids.shape[1], self.regressor.output_dim),
            fill_value=UNK_TOKEN,
            device=mol_emb.device,
            dtype=mol_emb.dtype
        )

        reg = torch.full(
            (x_ids.shape[0], x_ids.shape[1], self.regressor.output_dim),
            fill_value=UNK_TOKEN,
            device=mol_emb.device,
            dtype=mol_emb.dtype
        )

        # Create mask for valid indices
        valid_mask = (mol_emb != UNK_TOKEN).all(dim=-1)

        for i in range(mol_emb.shape[1]):  # iterate over components
            valid_indices = valid_mask[:, i]
            if valid_indices.any():
                # Apply regressor only to valid components
                clas[valid_indices, i, :], reg[valid_indices, i, :] = self.regressor(
                    mol_emb[valid_indices, i, :]
                )
        return (clas, reg)