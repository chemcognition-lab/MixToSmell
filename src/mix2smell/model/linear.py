import sys
import torch
import torch.nn as nn

from typing import Tuple


class FullyConnectedNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        super(FullyConnectedNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, x_ids, device):
        output = self.layers(x)
        return output