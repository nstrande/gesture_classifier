from __future__ import annotations

import json
import os
from importlib import import_module
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from src.config import config
from src.models.utils import BalancedJSONDataset

MODEL = import_module(config.model).Model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(
    dataset: Dataset, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare train, validation, and test data loaders.

    Args:
        dataset: The full dataset to split.
        batch_size: Batch size for the data loaders.

    Returns:
        Tuple of train, validation, and test data loaders.
    """
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_model(
    input_size: int, hidden_sizes: List[int], num_classes: int
) -> nn.Module:
    """Create and return the neural network model."""
    return MODEL(input_size, hidden_sizes, num_classes)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        device: Device to run the training on.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the model and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_acc: float,
    filename: str,
) -> None:
    """Save a checkpoint of the model."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }
    torch.save(checkpoint, filename)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
) -> Tuple[nn.Module, int]:
    """
    Train the model with early stopping.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        device: Device to run the training on.
        num_epochs: Maximum number of epochs to train.
        patience: Number of epochs to wait for improvement before early stopping.

    Returns:
        Tuple of the trained model and the number of epochs trained.
    """
    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        mlflow.log_metrics(
            {"train_loss": epoch_loss, "val_accuracy": val_acc}, step=epoch
        )

        checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_filename)
        mlflow.log_artifact(checkpoint_filename)
        os.remove(checkpoint_filename)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, "models/best_model.pt")
            mlflow.log_artifact("models/best_model.pt", "best_model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            mlflow.log_param("early_stopping_epoch", epoch + 1)
            break

    return model, epoch + 1


def main() -> None:
    """Main function to train and evaluate the neural network model."""
    set_seed(42)

    full_dataset: Dataset = BalancedJSONDataset("data/train_data")
    batch_size = 32
    train_loader, val_loader, test_loader = prepare_data(full_dataset, batch_size)

    num_classes = len(set(full_dataset.labels))
    input_size = 21 * 3
    hidden_sizes: List[int] = [128, 64, 32]

    model = create_model(input_size, hidden_sizes, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    mlflow.set_experiment("Classification_NN_Training_with_Checkpoints")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "input_size": input_size,
                "hidden_sizes": hidden_sizes,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "optimizer": type(optimizer).__name__,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
            }
        )

        model, epochs_trained = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=100,
            patience=10,
        )

        print("Training finished")

        test_acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.4f}")
        mlflow.log_metric("test_accuracy", test_acc)

        final_model_info: Dict[str, Any] = {
            "model": model,
            "epochs_trained": epochs_trained,
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc,
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "num_classes": num_classes,
            "label_to_idx": full_dataset.label_to_idx,
        }
        torch.save(final_model_info, "models/final_model.pt")
        mlflow.log_artifact("models/final_model.pt", "final_model")

        with open("models/label_to_idx.json", "w") as f:
            json.dump(full_dataset.label_to_idx, f)
        mlflow.log_artifact("models/label_to_idx.json")


if __name__ == "__main__":
    main()
