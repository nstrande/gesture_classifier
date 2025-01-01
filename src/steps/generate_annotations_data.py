from __future__ import annotations

import json
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from src.config import config
from src.utils.webcam import Webcam


class GestureAnnotator:
    def __init__(self, gesture_label: str) -> None:
        self.gesture_label: str = gesture_label.lower()
        self.sequence_buffer: list[dict] = []
        self.is_recording: bool = False
        self.sequence_count: int = 0
        self.prev_frame_time = 0
        self.frames_per_second = config.FRAMES_PER_SECOND
        self.sequence_length = config.SEQUENCE_LENGTH
        self.recording_duration = self.sequence_length / self.frames_per_second
        self.start_time = None

        # MediaPipe setup optimized for mobile devices
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Setup storage
        self.sequence_folder = f"data/raw/{gesture_label}"
        os.makedirs(self.sequence_folder, exist_ok=True)

    def toggle_recording(self) -> None:
        """Start a new recording or stop current recording."""
        if not self.is_recording:
            # Start new recording
            self.is_recording = True
            self.start_time = time.time()
            self.sequence_buffer = []
            print("Recording started")
        else:
            print("Recording already in progress")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process video frame and record landmarks if active."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if self.is_recording:
            current_time = time.time()
            elapsed_time = current_time - self.start_time

            # Check if recording should stop
            if elapsed_time >= self.recording_duration:
                if self.sequence_buffer:
                    self._save_sequence()
                self.is_recording = False
                self.sequence_buffer = []
                print("Recording completed")

        self._draw_status(frame, elapsed_time if self.is_recording else None)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=4
                ),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

            if self.is_recording:
                self._add_to_buffer(landmarks, frame)

        return frame

    def _draw_status(self, frame: np.ndarray, elapsed_time: float | None) -> None:
        """Draw recording status and countdown."""
        # Calculate FPS
        current_time = time.time()
        fps = (
            1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        )
        self.prev_frame_time = current_time

        # Colors
        recording_color = (0, 255, 0) if self.is_recording else (0, 0, 255)
        white = (255, 255, 255)

        # Left side information
        cv2.putText(
            frame,
            "Status: " + ("Recording" if self.is_recording else "Ready"),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            recording_color,
            2,
        )
        cv2.putText(
            frame,
            f"Gesture: {self.gesture_label}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            white,
            2,
        )
        cv2.putText(
            frame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2
        )
        cv2.putText(
            frame,
            f"Recorded Sequences: {self.sequence_count}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            white,
            2,
        )

        # Right side instructions
        right_margin = frame.shape[1] - 300
        cv2.putText(
            frame,
            "Controls:",
            (right_margin, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            white,
            2,
        )
        cv2.putText(
            frame,
            "R - Start Recording",
            (right_margin, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            white,
            2,
        )
        cv2.putText(
            frame,
            "Q - Quit",
            (right_margin, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            white,
            2,
        )

        # Show countdown if recording
        if self.is_recording and elapsed_time is not None:
            remaining_time = max(0, self.recording_duration - elapsed_time)
            cv2.putText(
                frame,
                f"Recording: {remaining_time:.1f}s",
                (right_margin, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                recording_color,
                2,
            )
        else:
            cv2.putText(
                frame,
                "Press R to start recording",
                (right_margin, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                recording_color,
                2,
            )

    def _add_to_buffer(
        self,
        landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList,
        frame: np.ndarray,
    ) -> None:
        """Add landmarks to current sequence buffer using pixel coordinates."""
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        landmark_list = []
        for lm in landmarks.landmark:
            pixel_x = int(lm.x * frame_width)
            pixel_y = int(lm.y * frame_height)
            # Keep z as is since it's relative depth
            landmark_list.append({"x": pixel_x, "y": pixel_y, "z": lm.z})

        self.sequence_buffer.append(landmark_list)

    def _save_sequence(self) -> None:
        """Save completed sequence to file."""
        sequence_path = os.path.join(
            self.sequence_folder, f"sequence_{self.sequence_count}.json"
        )
        with open(sequence_path, "w") as f:
            json.dump(self.sequence_buffer, f)
        self.sequence_count += 1


def main():
    """Run the gesture annotation tool."""
    gesture_label = input("Enter gesture label: ")
    annotator = GestureAnnotator(gesture_label)
    webcam = Webcam()

    print(f"\nRecording gestures for: {gesture_label}")
    print(f"Each recording will be {annotator.recording_duration:.1f} seconds")
    print("Press R to start recording")
    print("Press Q to quit\n")

    while True:
        frame = webcam.read_frame()
        if frame is None:
            break

        frame = annotator.process_frame(frame)
        cv2.imshow("Gesture Annotation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r") and not annotator.is_recording:
            annotator.toggle_recording()

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
