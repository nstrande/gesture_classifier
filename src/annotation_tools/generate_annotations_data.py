from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np

from src.utils.webcam import Webcam


class GestureAnnotator:
    """
    A class for annotating hand gestures in video frames using MediaPipe.

    Attributes:
        gesture_label (str): The label for the gesture being annotated.
        frame_count (int): Counter for the number of frames processed.
        mp_hands (mp.solutions.hands): MediaPipe hands solution object.
        hands (mp.solutions.hands.Hands): MediaPipe Hands object for processing hand landmarks.
        mp_drawing (mp.solutions.drawing_utils): MediaPipe drawing utilities for drawing landmarks.
        image_folder_path (str): Path to the folder where annotated images will be saved.
        landmarks_folder_path (str): Path to the folder where hand landmark data will be saved.
    """

    def __init__(self, gesture_label: str) -> None:
        """
        Initialize the GestureAnnotator with a gesture label and set up directories for saving annotations.

        Args:
            gesture_label (str): The label for the gesture being annotated.
        """
        self.gesture_label: str = gesture_label.lower()
        self.frame_count: int = 0
        self.mp_hands: mp.solutions.hands = mp.solutions.hands
        self.hands: mp.solutions.hands.Hands = self.mp_hands.Hands()
        self.mp_drawing: mp.solutions.drawing_utils = mp.solutions.drawing_utils
        self.image_folder_path: str = f"data/annotations/images/{gesture_label}"
        self.landmarks_folder_path: str = f"data/annotations/text_files/{gesture_label}"
        os.makedirs(self.image_folder_path, exist_ok=True)
        os.makedirs(self.landmarks_folder_path, exist_ok=True)

    def generate_annotations(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate annotations for a given frame by detecting hand landmarks.

        This method processes the input frame to detect hand landmarks using the MediaPipe Hands solution.
        If hand landmarks are detected, it draws the landmarks on the frame and saves the annotations.

        Args:
            frame (np.ndarray): The input frame in BGR format.

        Returns:
            np.ndarray: The frame with hand landmarks drawn on it.
        """
        rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result: mp.solutions.hands.Hands.process = self.hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                self._save_annotations(frame, hand_landmarks)
                print(f"Frame {self.frame_count} generated.")
        return frame

    def _save_annotations(
        self, frame: np.ndarray, hand_landmarks: mp.solutions.hands.HandLandmark
    ) -> None:
        """
        Save the current frame and its corresponding hand landmarks to disk.

        Args:
            frame (np.ndarray): The current video frame to be saved as an image.
            hand_landmarks (mp.solutions.hands.HandLandmark): The hand landmarks detected in the current frame.

        Saves:
            - The frame as an image file in the directory specified by `self.image_folder_path`.
            - The hand landmarks as a text file in the directory specified by `self.landmarks_folder_path`.
        """
        cv2.imwrite(f"{self.image_folder_path}/{self.frame_count}.png", frame)
        with open(f"{self.landmarks_folder_path}/{self.frame_count}.txt", "w") as f:
            for landmark in hand_landmarks.landmark:
                f.write(f"{landmark.x},{landmark.y},{landmark.z}\n")
        self.frame_count += 1


class HandGestureRecognition:
    """
    A class to handle hand gesture recognition using a webcam and an annotator.

    Attributes:
        webcam (Webcam): An instance of the Webcam class to capture video frames.
        annotator (GestureAnnotator): An instance of the GestureAnnotator class to generate annotations on video frames.
    """

    def __init__(self, webcam: Webcam, annotator: GestureAnnotator):
        """
        Initialize the HandGestureRecognition with a webcam and an annotator.

        Args:
            webcam (Webcam): An instance of the Webcam class to capture video frames.
            annotator (GestureAnnotator): An instance of the GestureAnnotator class to generate annotations on video frames.
        """
        self.webcam: Webcam = webcam
        self.annotator: GestureAnnotator = annotator

    def startup_countdown(self, seconds: int = 3) -> None:
        """
        Display a countdown on the webcam feed before starting the main process.

        Args:
            seconds (int): The number of seconds for the countdown. Defaults to 3.

        The method captures frames from the webcam, overlays a countdown text on each frame,
        and displays the frames in a window titled "Landmark Recognition". The countdown
        decreases by one second for each frame displayed. After the countdown reaches zero,
        it prints "Go!" to the console.
        """
        for i in range(seconds, 0, -1):
            frame: np.ndarray = self.webcam.read_frame()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                frame,
                f"Starting in {i} seconds...",
                (10, 30),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Landmark Recognition", frame)
            cv2.waitKey(1000)
        print("Go!")

    def run(self) -> None:
        """
        Run the annotation tool.

        This method starts with a countdown, then continuously reads frames from the webcam,
        generates annotations on each frame, and displays the annotated frames in a window.
        The loop runs until the user presses 'q' or the spacebar to exit.

        The annotated frames include a message instructing the user on how to exit.
        """
        self.startup_countdown()
        while True:
            frame: np.ndarray = self.webcam.read_frame()
            frame = self.annotator.generate_annotations(frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                frame,
                "Press 'q' or spacebar to exit",
                (10, 30),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Hand Landmark Recognition", frame)
            key: int = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord(" "):
                break
        self.webcam.release()
        cv2.destroyAllWindows()


def main() -> None:
    """
    Main function to run the gesture annotation tool.

    This function prompts the user to enter a gesture label, initializes the
    webcam, gesture annotator, and hand gesture recognition components, and
    starts the gesture recognition process.
    """
    gesture_label: str = input("Enter the gesture label: ")
    webcam: Webcam = Webcam()
    annotator: GestureAnnotator = GestureAnnotator(gesture_label)
    recognizer: HandGestureRecognition = HandGestureRecognition(webcam, annotator)
    recognizer.run()


if __name__ == "__main__":
    main()
