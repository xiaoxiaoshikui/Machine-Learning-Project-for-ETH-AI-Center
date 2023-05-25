"""Module for instantiating a neural network."""
import torch
from torch import Tensor, nn


class Network(nn.Module):
    def __init__(
        self,
        layer_num,
        input_dim,
        output_dim,
        hidden_dim,
        activation_function=torch.relu,
        last_activation=None,
    ):
        super().__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)
        layers = [
            nn.Linear(layers_unit[idx], layers_unit[idx + 1])
            for idx in range(len(layers_unit) - 1)
        ]
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1], output_dim)
        self.network_init()

    def forward(self, data: Tensor):
        return self._forward(data)

    def _forward(self, data: Tensor):
        for layer in self.layers:
            data = self.activation(layer(data))
        data = self.last_layer(data)
        if self.last_activation is not None:
            data = self.last_activation(data)
        return data

    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
