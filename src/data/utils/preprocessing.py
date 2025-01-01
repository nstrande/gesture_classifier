import mediapipe as mp
import torch


class LandmarkProcessor:
    """Processes hand landmarks for both training and inference."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame) -> list[dict[str, float]] | None:
        """Process a single frame and return landmarks if detected."""
        results = self.hands.process(frame)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks.landmark]
        return None

    @staticmethod
    def sequence_to_tensor(
        landmark_sequence: list[list[dict[str, float]]],
    ) -> torch.Tensor:
        """Convert a sequence of landmarks to model input tensor format."""
        processed_frames = []

        for frame in landmark_sequence:
            # Flatten x,y,z coordinates
            frame_coords = []
            for landmark in frame:
                frame_coords.extend([landmark["x"], landmark["y"], landmark["z"]])

            # Convert to tensor
            frame_tensor = torch.tensor(frame_coords, dtype=torch.float32)
            processed_frames.append(frame_tensor)

        # Stack all frames
        return torch.stack(processed_frames)

    def __del__(self):
        self.hands.close()
