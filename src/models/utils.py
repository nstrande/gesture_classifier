from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict
from typing import List
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BalancedJSONDataset(Dataset):
    """
    A dataset class that loads JSON data, balances the classes, and splits into train/test sets.
    """

    def __init__(self, data_dir: str, test_size: float = 0.2, random_state: int = 123):
        """
        Initialize the BalancedJSONDataset.

        Args:
            data_dir: Directory containing JSON files.
            test_size: Proportion of the dataset to include in the test split.
            random_state: Random state for reproducibility.
        """
        self.data: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.label_to_idx: Dict[str, int] = {}

        self._load_data(data_dir)
        self._encode_labels()
        self._balance_dataset()
        self._split_data(test_size, random_state)

    def _load_data(self, data_dir: str) -> None:
        """Load data from JSON files in the specified directory."""
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                with open(os.path.join(data_dir, filename), "r") as f:
                    json_data = json.load(f)
                    for key, value in json_data.items():
                        self.data.append(torch.tensor(value, dtype=torch.float32))
                        self.labels.append(key)

    def _encode_labels(self) -> None:
        """Convert string labels to numeric indices."""
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def _balance_dataset(self) -> None:
        """Balance the dataset by oversampling minority classes."""
        label_counts = Counter(self.labels)
        max_count = max(label_counts.values())

        balanced_data: List[torch.Tensor] = []
        balanced_labels: List[int] = []

        for label in label_counts:
            indices = [i for i, l in enumerate(self.labels) if l == label]
            balanced_indices = (
                indices * (max_count // len(indices))
                + indices[: max_count % len(indices)]
            )

            balanced_data.extend([self.data[i] for i in balanced_indices])
            balanced_labels.extend([label] * max_count)

        self.data = balanced_data
        self.labels = balanced_labels

    def _split_data(self, test_size: float, random_state: int) -> None:
        """Split the data into training and test sets."""
        X_train, _, y_train, _ = train_test_split(
            self.data,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels,
        )

        self.data = X_train
        self.labels = y_train

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing the data tensor and its corresponding label.
        """
        return self.data[idx], self.labels[idx]
