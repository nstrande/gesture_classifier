import json

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.config import config


class KalmanHandPredictor:
    def __init__(self, dt: float = 1 / 30):  # dt is time between frames, default 30 FPS
        # State: [x, y, z, dx/dt, dy/dt, dz/dt]
        self.state_dim = 6
        self.measurement_dim = 3

        # Initialize Kalman filter matrices
        self.state = np.zeros(self.state_dim)

        # State transition matrix F
        self.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement model matrix H
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

        # Process noise covariance matrix Q
        self.Q = np.eye(self.state_dim) * 0.1

        # Measurement noise covariance matrix R
        self.R = np.eye(self.measurement_dim) * 0.1

        # State covariance matrix P
        self.P = np.eye(self.state_dim)

        self.initialized = False

    def initialize(self, measurement: np.ndarray):
        """Initializes the Kalman filter with first measurement."""
        self.state[:3] = measurement
        self.state[3:] = 0  # Initialize velocity as 0
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Predicts next state."""
        # Update state
        self.state = self.F @ self.state

        # Update covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:3]  # Return only position

    def update(self, measurement: np.ndarray | None) -> np.ndarray:
        """Updates state with new measurement."""
        if measurement is None:
            # If no measurement, return prediction only
            return self.predict()

        if not self.initialized:
            self.initialize(measurement)
            return measurement

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        measurement_residual = measurement - (self.H @ self.state)
        self.state = self.state + (K @ measurement_residual)

        # Update covariance
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

        return self.state[:3]  # Return only position


class HandLandmarkPredictor:
    def __init__(self, dt: float = 1 / 30):
        self.predictors = {}  # A predictor for each landmark
        self.dt = dt
        self.missing_frames = 0

    def update_landmarks(self, landmarks: list[dict] | None) -> list[dict] | None:
        """Updates and predicts positions for all landmarks."""
        # If we've exceeded max extrapolation, reset predictors
        if self.missing_frames >= config.MAX_EXTRAPOLATION_FRAMES:
            self.predictors = {}
            self.missing_frames = 0
            return None

        if landmarks is None and not self.predictors:
            return None

        # If this is the first time we see landmarks, initialize predictors
        if landmarks is not None:
            self.predictors = {
                i: KalmanHandPredictor(self.dt) for i in range(len(landmarks))
            }
            self.missing_frames = 0
        else:
            self.missing_frames += 1

        predicted_landmarks = []
        for i in range(len(self.predictors)):
            if landmarks is None:
                # Prediction without measurement
                pos = self.predictors[i].update(None)
            else:
                # Update with new measurement
                lm = landmarks[i]
                measurement = np.array([lm["x"], lm["y"], lm["z"]])
                pos = self.predictors[i].update(measurement)

            predicted_landmarks.append(
                {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
            )

        return predicted_landmarks


class GestureRecognition:
    def __init__(self, model_path: str, label_path: str):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Initialize model
        self.device = torch.device("cpu")
        self.model_info = torch.load(model_path, map_location=self.device)

        # Load label mapping
        with open(label_path) as f:
            self.idx_to_label = {int(v): k for k, v in json.load(f).items()}

        # Setup model
        self.model = self.model_info["model"]
        self.model.load_state_dict(self.model_info["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Sequence buffer
        self.sequence_buffer = []
        self.max_seq_length = config.SEQUENCE_LENGTH

        # Kalman filter predictor
        self.landmark_predictor = HandLandmarkPredictor()

    def normalize_sequence(self, sequence: list[list[dict]]) -> list[list[dict]]:
        """Normalizes sequence relative to global min/max values."""
        if not sequence:
            return []

        # Find global min/max values
        x_values = [lm["x"] for frame in sequence for lm in frame]
        y_values = [lm["y"] for frame in sequence for lm in frame]
        z_values = [lm["z"] for frame in sequence for lm in frame]

        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        z_min, z_max = min(z_values), max(z_values)

        # Avoid division by zero
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        z_range = z_max - z_min if z_max != z_min else 1

        normalized_sequence = []
        for frame in sequence:
            normalized_frame = []
            for landmark in frame:
                normalized_frame.append(
                    {
                        "x": (landmark["x"] - x_min) / x_range,
                        "y": (landmark["y"] - y_min) / y_range,
                        "z": (landmark["z"] - z_min) / z_range,
                    }
                )
            normalized_sequence.append(normalized_frame)

        return normalized_sequence

    def preprocess_frame(self, frame):
        """Processes a frame and returns landmarks if a hand is detected."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        landmarks = None
        if results.multi_hand_landmarks:
            # Convert landmarks to our format
            frame_height, frame_width = frame.shape[:2]
            landmarks = [
                {"x": lm.x * frame_width, "y": lm.y * frame_height, "z": lm.z}
                for lm in results.multi_hand_landmarks[0].landmark
            ]

        # Use Kalman filter to update/predict landmarks
        predicted_landmarks = self.landmark_predictor.update_landmarks(landmarks)

        return predicted_landmarks, results

    def sequence_to_tensor(self, sequence: list[list[dict]]) -> torch.Tensor:
        """Converts normalized sequence to model input."""
        if not sequence:
            return None

        processed_frames = []
        for frame in sequence:
            frame_coords = []
            for landmark in frame:
                frame_coords.extend([landmark["x"], landmark["y"], landmark["z"]])
            processed_frames.append(torch.tensor(frame_coords, dtype=torch.float32))
        return torch.stack(processed_frames)

    def get_predictions(self, frame) -> tuple[dict[str, float], bool]:
        """Processes frame and returns predictions."""
        # Preprocess frame
        landmarks, results = self.preprocess_frame(frame)

        # If no hand is detected and no extrapolation is possible
        if landmarks is None:
            return {"No hand detected": 1.0}, results

        # Add landmarks to buffer
        self.sequence_buffer.append(landmarks)
        if len(self.sequence_buffer) > self.max_seq_length:
            self.sequence_buffer.pop(0)

        # If we don't have enough frames yet
        if len(self.sequence_buffer) < config.SEQUENCE_LENGTH:
            return {"Collecting data...": 1.0}, results

        # Normalize the sequence
        normalized_sequence = self.normalize_sequence(self.sequence_buffer)

        # Convert to tensor
        sequence = self.sequence_to_tensor(normalized_sequence)
        if sequence is None:
            return {"Error processing sequence": 1.0}, results

        sequence = sequence.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = self.model(sequence, torch.tensor([len(self.sequence_buffer)]))
            probabilities = torch.nn.functional.softmax(output, dim=1)

            probs_dict = {
                self.idx_to_label[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }

        return probs_dict, results


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    # Initialize gesture recognition
    gesture_recognition = GestureRecognition(
        model_path="models/gesture_classifier/final_model.pt",
        label_path="models/gesture_classifier/label_to_idx.json",
    )

    print(f"Sequence length: {config.SEQUENCE_LENGTH} frames")
    print(f"Recording at {config.FRAMES_PER_SECOND} FPS")
    print(f"Sequence duration: {config.SECONDS_PER_SEQUENCE} seconds")
    print(f"Max extrapolation frames: {config.MAX_EXTRAPOLATION_FRAMES}")

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from webcam")
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)

        # Get predictions and results
        predictions, results = gesture_recognition.get_predictions(frame)

        # Draw landmarks if any
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture_recognition.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    gesture_recognition.mp_hands.HAND_CONNECTIONS,
                    gesture_recognition.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=1
                    ),
                    gesture_recognition.mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2
                    ),
                )

        # Show buffer status and extrapolation status
        buffer_text = f"Buffer: {len(gesture_recognition.sequence_buffer)}/{gesture_recognition.max_seq_length}"
        cv2.putText(
            frame,
            buffer_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        extrap_text = f"Missing frames: {gesture_recognition.landmark_predictor.missing_frames}/{config.MAX_EXTRAPOLATION_FRAMES}"
        cv2.putText(
            frame,
            extrap_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Show predictions
        y_pos = 90
        if predictions:
            sorted_predictions = sorted(
                predictions.items(), key=lambda x: x[1], reverse=True
            )
            for gesture, confidence in sorted_predictions[:3]:
                color = (0, 255, 0) if confidence > 0.5 else (255, 255, 255)
                text = f"{gesture}: {confidence:.2f}"
                cv2.putText(
                    frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
                y_pos += 30

        # Show frame
        cv2.imshow("Gesture Recognition", frame)

        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
