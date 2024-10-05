from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from src.models.predict import load_model
from src.models.predict import predict
from src.utils.webcam import Webcam  # Import the Webcam class


def initialize_mediapipe() -> (
    Tuple[mp.solutions.hands, mp.solutions.drawing_utils, mp.solutions.hands.Hands]
):
    """Initialize MediaPipe components for hand detection."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands()
    return mp_hands, mp_drawing, hands


def initialize_model(
    model_path: Path,
) -> Tuple[nn.Module, Dict[str, int], torch.device]:
    """Load the model and related components."""
    model, label_to_idx, device, _ = load_model(model_path)
    print(f"Using device: {device}")
    return model, label_to_idx, device


def process_frame(
    frame: np.ndarray,
    hands: mp.solutions.hands.Hands,
    mp_drawing: mp.solutions.drawing_utils,
    mp_hands: mp.solutions.hands,
    model: nn.Module,
    label_to_idx: Dict[str, int],
    device: torch.device,
) -> np.ndarray:
    """Process a single frame for hand detection and gesture recognition."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        try:
            input_data = torch.tensor(
                [
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in hand_landmarks.landmark
                ],
                dtype=torch.float,
            )
            prediction = predict(model, input_data, label_to_idx, device)

            cv2.putText(
                frame,
                f"Gesture: {prediction}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        except Exception as e:
            print(f"Error during prediction: {e}")
    return frame


def main() -> None:
    """Run the hand gesture recognition system."""
    model_path = Path("models/final_model.pt")
    model, label_to_idx, device = initialize_model(model_path)
    mp_hands, mp_drawing, hands = initialize_mediapipe()

    webcam = Webcam()  # Initialize the Webcam class

    try:
        while True:
            frame = webcam.read_frame()
            frame = process_frame(
                frame, hands, mp_drawing, mp_hands, model, label_to_idx, device
            )
            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        webcam.release()  # Ensure the webcam is released
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
