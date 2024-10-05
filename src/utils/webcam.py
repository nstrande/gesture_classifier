from __future__ import annotations

import cv2


class Webcam:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam.")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read frame.")
        return frame

    def release(self):
        self.cap.release()
