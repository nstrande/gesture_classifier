from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout_rate: float = 0.5) -> None:
        """
        Initialize the LSTM-based classification model.

        Args:
            input_size: Size of input features.
            hidden_size: Number of neurons in LSTM hidden layers.
            num_layers: Number of LSTM layers.
            num_classes: Number of classification categories.
            dropout_rate: Dropout rate for regularization.
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the LSTM-based classification network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Class scores as a tensor.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the output of the last time step
        out = self.fc(out)
        return out
