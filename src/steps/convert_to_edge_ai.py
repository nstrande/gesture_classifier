from __future__ import annotations

import coremltools as ct
import torch
import torch.nn as nn

from src.main import GestureRecognition


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Input: (batch_size, 21 landmarks, 3 coordinates)
        batch_size = x.shape[0]

        # Reshape to match model's expected input of 63 features
        # Flattening 21 landmarks * 3 coordinates = 63 features
        x_reshaped = x.reshape(batch_size, -1)  # Shape: (batch_size, 63)

        # Add sequence dimension for LSTM: (batch_size, 1, 63)
        x_reshaped = x_reshaped.unsqueeze(1)

        # All sequences have length 1
        lengths = torch.ones(batch_size, dtype=torch.long, device=x.device)

        # Forward pass through model
        output = self.model(x_reshaped, lengths)

        return output


def convert_to_coreml(model_path: str, label_path: str, output_path: str):
    """
    Convert PyTorch LSTM model to CoreML format.
    """
    # Load model
    gesture_recognition = GestureRecognition(model_path, label_path)
    model = gesture_recognition.model
    idx_to_label = gesture_recognition.idx_to_label

    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    wrapped_model.to("cpu")

    # Input: (batch_size, 21 landmarks, 3 coordinates)
    dummy_input = torch.randn(1, 21, 3, device="cpu")

    # Test forward pass
    with torch.no_grad():
        try:
            test_output = wrapped_model(dummy_input)
            print(f"Test forward pass successful. Output shape: {test_output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            print(f"Input shape: {dummy_input.shape}")
            print(f"LSTM input size: {model.lstm.input_size}")
            print(f"LSTM hidden size: {model.lstm.hidden_size}")
            raise

    # Convert and save
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, dummy_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=dummy_input.shape, name="input_1")],
        classifier_config=ct.ClassifierConfig(class_labels=list(idx_to_label.values())),
    )

    mlmodel.save(output_path)
    print(f"Model saved to {output_path}")


def main():
    convert_to_coreml(
        model_path="models/gesture_classifier/final_model.pt",
        label_path="models/gesture_classifier/label_to_idx.json",
        output_path="models/handgesture_classifier.mlpackage",
    )


if __name__ == "__main__":
    main()
