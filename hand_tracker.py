"""
HoloDraw Hand Tracker Module
-----------------------------
Wraps the MediaPipe Tasks HandLandmarker API for real-time hand landmark
detection. This module uses the modern mp.tasks.vision API (MediaPipe 0.10+)
which requires a .task model file.

The HandLandmarker model runs a deep learning pipeline that:
    1. Detects palm regions using a lightweight detection network.
    2. Crops and feeds each palm to a landmark regression network.
    3. Outputs 21 3D landmarks per detected hand.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple

from config import (
    INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, THUMB_TIP,
)

# MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Default model path (bundled alongside the source files)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)


class HandTracker:
    """
    Detects hand landmarks in video frames using the MediaPipe HandLandmarker.

    This class handles frame preprocessing (BGR to RGB conversion, wrapping
    into mp.Image), landmark detection via the Tasks API in VIDEO mode,
    and coordinate conversion from normalized [0, 1] space to pixel space.

    Attributes:
        _landmarker: The MediaPipe HandLandmarker instance.
        _frame_timestamp_ms: Monotonically increasing timestamp for VIDEO mode.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):
        """
        Initialize the hand tracker with the MediaPipe HandLandmarker model.

        Args:
            model_path: Path to the hand_landmarker.task model file.
            max_hands: Maximum number of hands to detect simultaneously.
            detection_confidence: Minimum confidence for initial hand detection.
            tracking_confidence: Minimum confidence for landmark tracking.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"HandLandmarker model not found at: {model_path}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms: int = 0

    def detect(self, frame: np.ndarray) -> Optional[HandLandmarkerResult]:
        """
        Process a BGR frame and return hand detection results.

        The frame is converted from BGR (OpenCV) to RGB (MediaPipe) and wrapped
        in an mp.Image object. The VIDEO running mode requires monotonically
        increasing timestamps, which are tracked internally.

        Args:
            frame: A BGR image as a NumPy array (H x W x 3).

        Returns:
            HandLandmarkerResult if hands were detected, None otherwise.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_timestamp_ms += 33  # Approximate 30 FPS interval

        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if result.hand_landmarks:
            return result
        return None

    def get_landmark_list(
        self,
        result: HandLandmarkerResult,
        frame_shape: Tuple[int, int],
        hand_index: int = 0,
    ) -> List[Tuple[int, int]]:
        """
        Convert all 21 normalized landmarks to pixel coordinates for one hand.

        Args:
            result: The HandLandmarkerResult from detect().
            frame_shape: (height, width) of the frame used during detection.
            hand_index: Index of the hand in the results (default: 0 for first).

        Returns:
            A list of 21 (x, y) tuples in pixel coordinates, one per landmark.
        """
        h, w = frame_shape
        landmarks = result.hand_landmarks[hand_index]
        pixel_landmarks = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pixel_landmarks.append((px, py))
        return pixel_landmarks

    def get_fingertip_positions(
        self,
        result: HandLandmarkerResult,
        frame_shape: Tuple[int, int],
        hand_index: int = 0,
    ) -> dict:
        """
        Extract pixel positions for all five fingertips.

        Args:
            result: The HandLandmarkerResult from detect().
            frame_shape: (height, width) of the frame used during detection.
            hand_index: Index of the hand in the results (default: 0).

        Returns:
            A dictionary mapping fingertip names to (x, y) pixel coordinates:
            {"thumb", "index", "middle", "ring", "pinky"}.
        """
        lm_list = self.get_landmark_list(result, frame_shape, hand_index)
        return {
            "thumb": lm_list[THUMB_TIP],
            "index": lm_list[INDEX_TIP],
            "middle": lm_list[MIDDLE_TIP],
            "ring": lm_list[RING_TIP],
            "pinky": lm_list[PINKY_TIP],
        }

    def draw_landmarks_on_frame(
        self,
        frame: np.ndarray,
        result: HandLandmarkerResult,
    ) -> np.ndarray:
        """
        Draw detected hand landmarks and connections onto the frame.

        Uses the MediaPipe drawing utilities to render the hand skeleton.
        Useful for debugging and visual feedback during development.

        Args:
            frame: The BGR frame to draw on (modified in place).
            result: HandLandmarkerResult containing detected landmarks.

        Returns:
            The frame with landmarks drawn on it.
        """
        if result and result.hand_landmarks:
            drawing_utils = mp.tasks.vision.drawing_utils
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

            for hand_landmarks in result.hand_landmarks:
                # Convert NormalizedLandmark list to the format draw_landmarks expects
                proto_landmarks = mp.solutions.hands.HandLandmark if hasattr(mp, 'solutions') else None

                # Draw each landmark as a circle and each connection as a line
                h, w, _ = frame.shape
                for i, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 200, 0), -1, cv2.LINE_AA)

                # Draw connections between landmarks
                lm_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for connection in connections:
                    start_idx = connection.start
                    end_idx = connection.end
                    if start_idx < len(lm_pixels) and end_idx < len(lm_pixels):
                        cv2.line(
                            frame,
                            lm_pixels[start_idx],
                            lm_pixels[end_idx],
                            (0, 150, 0), 1, cv2.LINE_AA,
                        )
        return frame

    def release(self):
        """Release MediaPipe resources."""
        self._landmarker.close()
