from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Get the best available device for PyTorch operations."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(
    model_path: Path,
) -> Tuple[nn.Module, Dict[str, int], torch.device, Dict[str, Any]]:
    """
    Load a PyTorch model and its associated information.

    Returns: model, label_to_idx mapping, device, and full model info.
    """
    device = get_device()
    model_info = torch.load(model_path, map_location=device)

    model: nn.Module = model_info["model"]
    model.load_state_dict(model_info["model_state_dict"])
    model.to(device)
    model.eval()

    return model, model_info["label_to_idx"], device, model_info


def predict(
    model: nn.Module,
    input_data: torch.Tensor | list,
    label_to_idx: Dict[str, int],
    device: torch.device,
) -> str:
    """
    Make a prediction using the provided model and input data.

    Returns: Predicted label as a string.
    """
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)

    if input_data.dim() == 2:
        input_data = input_data.unsqueeze(0)  # Add batch dimension if missing

    input_data = input_data.to(device)

    with torch.no_grad():
        output = model(input_data)
        _, predicted = torch.max(output, 1)

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    return idx_to_label[predicted.item()]
