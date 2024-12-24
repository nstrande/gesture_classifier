from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x, lengths):
        # Pack sequence
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM forward pass
        packed_output, _ = self.lstm(packed_x)
        
        # Unpack output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Attention
        attention_weights = []
        for i, length in enumerate(lengths):
            seq_attention = self.attention(output[i, :length])
            padded_attention = torch.cat([
                seq_attention,
                torch.full((output.size(1) - length, 1), float('-inf'), device=seq_attention.device)
            ])
            attention_weights.append(padded_attention)
        
        attention_weights = torch.stack(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.bmm(attention_weights.transpose(1, 2), output)
        context = context.squeeze(1)
        
        # Classification
        output = self.fc(context)
        return output
