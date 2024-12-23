from __future__ import annotations

import json
import os
import time
from typing import Dict
from typing import List

import cv2
import mediapipe as mp
import numpy as np

from src.utils.webcam import Webcam


class GestureAnnotator:
    """
    Records dynamic hand gesture sequences.

    Records sequences of varying length using spacebar control.
    Provides real-time visual feedback during recording.

    Attributes:
        gesture_label (str): Label for recorded gesture
        sequence_buffer (List): Current sequence buffer
        is_recording (bool): Recording state
        sequence_count (int): Number of completed sequences
    """

    def __init__(self, gesture_label: str) -> None:
        self.gesture_label: str = gesture_label.lower()
        self.sequence_buffer: List[Dict] = []
        self.is_recording: bool = False
        self.sequence_count: int = 0
        self.prev_frame_time = 0

        # MediaPipe setup optimized for mobile
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Setup storage
        self.sequence_folder = f"data/annotations/{gesture_label}"
        os.makedirs(self.sequence_folder, exist_ok=True)

    def toggle_recording(self) -> None:
        """Toggle recording and save sequence if stopping."""
        if self.is_recording and self.sequence_buffer:
            self._save_sequence()
            self.sequence_buffer = []
        self.is_recording = not self.is_recording

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process video frame and record landmarks if active."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        self._draw_status(frame)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )

            if self.is_recording:
                self._add_to_buffer(landmarks)

        return frame

    def _draw_status(self, frame: np.ndarray) -> None:
        """
        Draw recording status and instructions.

        Layout:
        Left side:
            - Recording status
            - Current gesture
            - FPS
            - Sequence count
        Right side:
            - Controls info
            - Recording instructions
        """
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = current_time

        # Colors
        recording_color = (0, 255, 0) if self.is_recording else (0, 0, 255)
        white = (255, 255, 255)

        # Left side information
        cv2.putText(frame, "Status: " + ("Recording" if self.is_recording else "Ready"),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
        cv2.putText(frame, f"Gesture: {self.gesture_label}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
        cv2.putText(frame, f"Recorded Sequences: {self.sequence_count}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)

        # Right side instructions
        right_margin = frame.shape[1] - 300
        cv2.putText(frame, "Controls:",
                    (right_margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
        cv2.putText(frame, "SPACE - Start/Stop Recording",
                    (right_margin, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
        cv2.putText(frame, "Q - Quit",
                    (right_margin, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)

        # Recording instructions
        if self.is_recording:
            cv2.putText(frame, "Press SPACE to stop recording",
                        (right_margin, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
        else:
            cv2.putText(frame, "Press SPACE to start recording",
                        (right_margin, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)

    def _add_to_buffer(self, landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> None:
        """Add landmarks to current sequence buffer."""
        landmark_list = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks.landmark]
        self.sequence_buffer.append(landmark_list)

    def _save_sequence(self) -> None:
        """Save completed sequence to file."""
        sequence_path = os.path.join(
            self.sequence_folder,
            f"sequence_{self.sequence_count}.json"
        )
        with open(sequence_path, 'w') as f:
            json.dump(self.sequence_buffer, f)
        self.sequence_count += 1

def main():
    """Run the gesture annotation tool."""
    gesture_label = input("Enter gesture label: ")
    annotator = GestureAnnotator(gesture_label)
    webcam = Webcam()

    print(f"\nRecording gestures for: {gesture_label}")
    print("Press SPACE to start/stop recording")
    print("Press Q to quit\n")

    while True:
        frame = webcam.read_frame()
        if frame is None:
            break

        frame = annotator.process_frame(frame)
        cv2.imshow("Gesture Annotation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            annotator.toggle_recording()

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
