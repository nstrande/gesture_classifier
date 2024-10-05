from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Flexible neural network for classification tasks.
    Customizable architecture with dropout and batch normalization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize the classification model.

        Args:
            input_size: Size of input features.
            hidden_sizes: List of neurons in each hidden layer.
            num_classes: Number of classification categories.
            dropout_rate: Dropout rate for regularization.
        """
        super(Model, self).__init__()
        self.flatten = nn.Flatten()

        layers: List[nn.Module] = []
        prev_size: int = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the classification network.

        Args:
            x: Input tensor.

        Returns:
            Class scores as a tensor.
        """
        x = self.flatten(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
