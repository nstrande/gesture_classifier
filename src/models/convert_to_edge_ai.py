from __future__ import annotations

from pathlib import Path

import coremltools as ct
import torch

from src.models.predict import load_model


def convert_to_coreml(model_path: str, output_path: str):
    """
    Convert PyTorch model to CoreML format.
    """
    # Load model
    model, label_to_idx, device, _ = load_model(Path(model_path))

    # Create reverse mapping
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Ensure model is in eval mode and on CPU
    model.eval()
    model.to("cpu")

    # Create dummy input
    dummy_input = torch.randn(1, 21, 3, device="cpu")

    # Trace model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=dummy_input.shape, name="input_1")],
        classifier_config=ct.ClassifierConfig(class_labels=list(idx_to_label.values())),
    )

    # Save model
    mlmodel.save(output_path)
    print(f"Model saved to {output_path}")


def main():
    convert_to_coreml(
        model_path="models/final_model.pt",
        output_path="models/handgesture_classifier.mlpackage",
    )


if __name__ == "__main__":
    main()
