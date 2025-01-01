from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with cosine annealing and warmup.
    Particularly suitable for LSTM training:
    1. Warmup period helps establish good initial memory patterns
    2. Cosine decay provides smooth learning rate reduction
    3. Helps prevent gradient issues common in LSTM training
    """

    def __init__(
        self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, warmup_start_lr=1e-7
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_max_lr = optimizer.param_groups[0]["lr"]  # Store initial lr as max
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (self.base_max_lr - self.warmup_start_lr)
                for _ in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (self.base_max_lr - self.min_lr) * cosine_decay
                for _ in self.base_lrs
            ]


class Model(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int,
        hidden_size: int,
        num_layers=2,
        dropout=0.5,
        learning_rate=0.001,
        num_epochs=50,  # Added to configure scheduler
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # Initialize optimizer and scheduler
        self.configure_optimizers(learning_rate)

    def configure_optimizers(self, learning_rate: float) -> None:
        """
        Initialize optimizer with cosine warmup scheduler optimized for LSTM
        """
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),  # Default Adam betas
            eps=1e-8,  # Default Adam epsilon
            weight_decay=0.01,  # Small weight decay for regularization
        )

        # Configure scheduler with warmup
        warmup_epochs = 3  # Short warmup for LSTM
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=self.num_epochs,
            min_lr=learning_rate / 100,  # Minimum LR is 1% of max
            warmup_start_lr=learning_rate / 10,  # Start at 10% of max
        )

    def get_optimizer_info(self) -> dict[str, any]:
        """
        Return optimizer and scheduler info for logging
        """
        return {
            "optimizer_name": self.optimizer.__class__.__name__,
            "scheduler_name": "CosineWarmupScheduler",
            "initial_lr": self.optimizer.param_groups[0]["lr"],
            "warmup_epochs": self.scheduler.warmup_epochs,
            "min_lr": self.scheduler.min_lr,
            "warmup_start_lr": self.scheduler.warmup_start_lr,
        }

    def optimizer_step(self, loss: torch.Tensor) -> None:
        """
        Perform optimization step
        """
        self.optimizer.zero_grad()
        loss.backward()

        # Optional: Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

    def scheduler_step(self, val_loss: float = None) -> float:
        """
        Step the scheduler. This scheduler doesn't use validation loss.
        """
        self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]

    def forward(self, x, lengths):
        # Pack sequence
        packed_x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # LSTM forward pass
        packed_output, _ = self.lstm(packed_x)

        # Unpack output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Attention
        attention_weights = []
        for i, length in enumerate(lengths):
            seq_attention = self.attention(output[i, :length])
            padded_attention = torch.cat(
                [
                    seq_attention,
                    torch.full(
                        (output.size(1) - length, 1),
                        float("-inf"),
                        device=seq_attention.device,
                    ),
                ]
            )
            attention_weights.append(padded_attention)

        attention_weights = torch.stack(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention
        context = torch.bmm(attention_weights.transpose(1, 2), output)
        context = context.squeeze(1)

        # Classification
        output = self.fc(context)
        return output
