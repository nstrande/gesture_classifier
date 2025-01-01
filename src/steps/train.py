import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from src.models.model_config.classification_lstm import Model as MODEL
from src.models.utils import JSONDataset
from src.utils.plotting import confusion_matrix_plot

NUM_EPOCHS = 25


def collate_fn(batch):
    """
    Custom collate function to be used with a DataLoader for batching sequences of varying lengths.

    Args:
        batch (list of tuples): A list of tuples where each tuple contains a sequence and its corresponding label.
                                Example: [(sequence1, label1), (sequence2, label2), ...]

    Returns:
        tuple: A tuple containing:
            - padded_seqs (Tensor): A tensor of shape (batch_size, max_seq_length) containing the padded sequences.
            - labels (Tensor): A tensor of shape (batch_size,) containing the labels.
            - lengths (Tensor): A tensor of shape (batch_size,) containing the lengths of the original sequences.
    """
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True)
    return padded_seqs, torch.LongTensor(labels), lengths


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.

    Attributes:
        patience (int): Number of epochs to wait after last time validation loss improved.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        counter (int): Counter for the number of epochs with no improvement.
        best_loss (float or None): Best recorded validation loss.
        early_stop (bool): Flag to indicate whether training should be stopped.
    """

    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if validation loss has improved and update early stopping status.

        Args:
            val_loss (float): Current epoch's validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class TrainModel:
    """
    A class used to train a machine learning model for gesture classification.

    Attributes:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_epochs (int): Number of epochs to train the model.
        patience (int): Number of epochs to wait for improvement before early stopping.
        min_delta (float): Minimum change in monitored quantity to qualify as improvement.
        checkpoint_frequency (int): How often to save checkpoints (in epochs).
        save_dir (Path): Directory to save model artifacts.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        label_to_idx,
        num_epochs=NUM_EPOCHS,
        patience=3,
        min_delta=0.001,
        checkpoint_frequency=5,
        save_dir="models/gesture_classifier/",
    ):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        self.checkpoint_frequency = checkpoint_frequency
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.label_to_idx = label_to_idx

        # Initialize MLflow
        mlflow.set_experiment("Sign Language Recognition")

        # Get optimizer info from model for logging
        optimizer_info = self.model.get_optimizer_info()

        # Log model parameters
        self.run = mlflow.start_run()
        mlflow.log_params(
            {
                **optimizer_info,
                "num_epochs": num_epochs,
                "model": self.model.__class__.__name__,
                "device": self.device,
                "criterion": self.criterion.__class__.__name__,
                "early_stopping_patience": patience,
                "early_stopping_min_delta": min_delta,
                "checkpoint_frequency": checkpoint_frequency,
            }
        )

    def save_checkpoint(
        self, epoch: int, train_loss: float, val_loss: float, val_accuracy: float
    ) -> str:
        """
        Save a training checkpoint to resume training later.

        Args:
            epoch: Current epoch number
            train_loss: Current training loss
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy

        Returns:
            str: Path to saved checkpoint
        """
        checkpoint_dir = self.save_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "scheduler_state_dict": self.model.scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    def save_model(
        self,
        epochs_trained: int,
        val_accuracy: float,
        test_accuracy: float,
        is_best: bool = False,
    ) -> tuple[str, str]:
        """
        Save model with all necessary information for inference.

        Args:
            epochs_trained: Total number of epochs trained
            val_accuracy: Final validation accuracy
            test_accuracy: Final test accuracy
            is_best: Whether this is the best performing model
        """
        # Get original dataset from random split
        dataset = self.train_loader.dataset.dataset

        model_info: dict[str, any] = {
            "model": self.model,
            "model_state_dict": self.model.state_dict(),
            "epochs_trained": epochs_trained,
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "scheduler_state_dict": self.model.scheduler.state_dict(),
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "is_best": is_best,
            "model_config": {
                "input_size": self.model.lstm.input_size,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "num_classes": len(dataset.label_to_idx),
            },
            "label_to_idx": dataset.label_to_idx,
        }

        # Save model
        model_path = self.save_dir / "final_model.pt"
        torch.save(model_info, model_path)

        label_path = self.save_dir / "label_to_idx.json"
        with open(label_path, "w") as f:
            json.dump(dataset.label_to_idx, f, indent=4)

        return str(model_path), str(label_path)

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            tuple: (average train loss, training accuracy)
        """
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target, lengths in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            lengths = lengths.to(self.device)

            output = self.model(data, lengths)
            loss = self.criterion(output, target)

            # Use model's optimizer step
            self.model.optimizer_step(loss)

            train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_accuracy = 100 * correct / total
        return train_loss / len(self.train_loader), train_accuracy

    def evaluate(self, data_loader):
        """
        Evaluate model on given dataset.

        Args:
            data_loader: DataLoader for the dataset to evaluate on

        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target, lengths in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                lengths = lengths.to(self.device)

                output = self.model(data, lengths)
                total_loss += self.criterion(output, target).item()

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def save_confusion_matrix(self, dataloader):
        """Generate confusion matrix plot and save it to MLflow."""

        y_true = []
        y_pred = []

        with torch.no_grad():
            for data, target, lengths in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                lengths = lengths.to(self.device)

                output = self.model(data, lengths)
                _, predicted = torch.max(output.data, 1)

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        fig = confusion_matrix_plot(
            y_true,
            y_pred,
            labels=self.label_to_idx.keys(),
            title="Confusion Matrix",
            normalize=True,
        )
        mlflow.log_figure(fig, "confusion_matrix.png")

    def train(self):
        """
        Trains the model for a specified number of epochs, logs metrics to MLflow,
        saves the best model, and implements early stopping.

        The training process includes:
        - Training the model for each epoch and evaluating it on validation data.
        - Logging training and validation metrics (loss, accuracy, learning rate) to MLflow.
        - Adjusting the learning rate using the model's scheduler.
        - Saving the model with the best validation accuracy.
        - Implementing early stopping based on validation loss.
        - Saving periodic checkpoints for training resumption.
        - Saving final model with all necessary information.
        - Final evaluation on test set.
        """
        best_val_acc = 0
        current_lr = self.model.optimizer.param_groups[0]["lr"]

        print(f"Training started on device: {self.device}")

        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.evaluate(self.val_loader)

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            # Save checkpoint if needed
            if (epoch + 1) % self.checkpoint_frequency == 0:
                checkpoint_path = self.save_checkpoint(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                )
                mlflow.log_artifact(checkpoint_path, "checkpoints")

            # Learning rate scheduling
            new_lr = self.model.scheduler_step(val_loss)
            if new_lr != current_lr:
                current_lr = new_lr
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # Save best model state
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_state = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.model.optimizer.state_dict(),
                    "scheduler_state_dict": self.model.scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "val_accuracy": val_accuracy,
                }

            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                mlflow.log_metric("stopped_epoch", epoch + 1)
                break

            print(f"Epoch: {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.2f}%")
            print(f"Learning Rate: {current_lr}")
            print("-" * 50)

        # Load best model state
        self.model.load_state_dict(best_model_state["model_state_dict"])
        self.model.optimizer.load_state_dict(best_model_state["optimizer_state_dict"])
        self.model.scheduler.load_state_dict(best_model_state["scheduler_state_dict"])

        # Final evaluation on test set with best model
        final_test_loss, final_test_accuracy = self.evaluate(self.test_loader)
        mlflow.log_metrics(
            {
                "final_test_loss": final_test_loss,
                "final_test_accuracy": final_test_accuracy,
            }
        )

        # Save confusion matrix plot
        self.save_confusion_matrix(self.test_loader)

        # Save final model (which is the best model)
        self.save_model(
            epochs_trained=best_model_state["epoch"],
            val_accuracy=best_model_state["val_accuracy"],
            test_accuracy=final_test_accuracy,
            is_best=False,
        )

        print("\nFinal Test Results (Best Model):")
        print(f"Test Loss: {final_test_loss:.4f}")
        print(f"Test Accuracy: {final_test_accuracy:.2f}%")

        mlflow.end_run()


def main():
    # Data setup
    data_dir = "data/processed"
    dataset = JSONDataset(data_dir)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.1 * total_size)  # 10% for validation
    test_size = total_size - train_size - val_size  # Remaining 20% for testing

    # Split data
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # Initialize model
    model = MODEL(
        num_classes=len(dataset.label_to_idx),
        input_size=63,  # 21 joints * 3 dimensions that is input from MediaPipe Hand landmarks
        hidden_size=128,  # LSTM hidden size
    )

    # Train model with MLflow tracking
    trainer = TrainModel(
        model, train_loader, val_loader, test_loader, label_to_idx=dataset.label_to_idx
    )
    trainer.train()


if __name__ == "__main__":
    main()
