"""
HoloDraw Gesture Recognizer Module
------------------------------------
Classifies hand gestures from MediaPipe landmark data using geometric rules.
Each gesture is determined by analyzing which fingers are extended and the
spatial relationships between specific landmark points.

Gesture Definitions:
    DRAW     - Index finger extended, middle/ring/pinky curled (thumb ignored).
    NAVIGATE - Index and middle fingers extended, ring/pinky curled (thumb ignored).
    SELECT   - Thumb tip and index tip are close together (pinch).
    UNDO     - All fingers curled including thumb (fist).
    REDO     - Only the thumb is extended, all others curled (thumbs up).
    IDLE     - Any other hand configuration (e.g., open palm).
"""

import math
from typing import List, Tuple, Optional

from config import (
    Gesture,
    THUMB_TIP, THUMB_IP, THUMB_MCP,
    INDEX_TIP, INDEX_PIP, INDEX_MCP,
    MIDDLE_TIP, MIDDLE_PIP,
    RING_TIP, RING_PIP,
    PINKY_TIP, PINKY_PIP,
    WRIST,
    PINCH_DISTANCE_THRESHOLD,
    GESTURE_DEBOUNCE_FRAMES,
    UNDO_REDO_COOLDOWN_FRAMES,
)


class GestureRecognizer:
    """
    Classifies hand gestures from a list of 21 hand landmark positions.

    Uses geometric heuristics (finger extension checks, pinch distance) rather
    than a separate ML classifier. Includes debouncing logic to prevent
    erratic gesture switches caused by momentary misdetections.

    Attributes:
        _prev_gesture: The last accepted gesture after debouncing.
        _candidate_gesture: The gesture currently being evaluated for acceptance.
        _candidate_count: Number of consecutive frames the candidate has persisted.
        _undo_redo_cooldown: Remaining cooldown frames after an undo/redo trigger.
    """

    def __init__(self):
        """Initialize the gesture recognizer with default state."""
        self._prev_gesture: Gesture = Gesture.IDLE
        self._candidate_gesture: Gesture = Gesture.IDLE
        self._candidate_count: int = 0
        self._undo_redo_cooldown: int = 0

    def classify(self, landmarks: List[Tuple[int, int]]) -> Gesture:
        """
        Determine the current gesture from hand landmark positions.

        The classification pipeline:
        1. Determine which fingers are extended.
        2. Check for pinch gesture, but only if the index finger is at least
           partially extended (guards against fist being detected as pinch).
        3. Map finger states to a gesture.
        4. Apply debouncing to stabilize the output.

        Args:
            landmarks: List of 21 (x, y) pixel coordinates from HandTracker.

        Returns:
            The debounced Gesture enum value.
        """
        if not landmarks or len(landmarks) < 21:
            return self._apply_debounce(Gesture.IDLE)

        finger_states = self.get_finger_states(landmarks)

        # Check pinch gesture (thumb and index tips close together).
        # Guard: only recognize a pinch if the index finger is at least partially
        # extended. In a fist, the thumb and index tips are close together but
        # the index is fully curled - that should be UNDO, not SELECT.
        index_extended = finger_states[1]
        if index_extended and self._is_pinch(landmarks):
            raw_gesture = Gesture.SELECT
        else:
            raw_gesture = self._map_finger_states_to_gesture(finger_states)

        return self._apply_debounce(raw_gesture)

    def get_finger_states(self, landmarks: List[Tuple[int, int]]) -> List[bool]:
        """
        Determine which fingers are extended (open) vs. curled (closed).

        For the four non-thumb fingers, a finger is considered extended if its
        tip is above (lower y-value in screen coords) its PIP joint. For the
        thumb, extension is checked using the x-axis distance from the palm
        center, accounting for both left and right hands.

        Args:
            landmarks: List of 21 (x, y) pixel coordinates.

        Returns:
            A list of 5 booleans: [thumb, index, middle, ring, pinky].
            True indicates the finger is extended.
        """
        states = []

        # Thumb: Compare tip x-position relative to the IP joint.
        # Determine hand orientation by checking if the thumb MCP is to the
        # left or right of the wrist to handle both hands correctly.
        is_right_hand = landmarks[THUMB_MCP][0] > landmarks[WRIST][0]
        if is_right_hand:
            # For a right hand (mirrored view), thumb extends to the right.
            states.append(landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0])
        else:
            # For a left hand (mirrored view), thumb extends to the left.
            states.append(landmarks[THUMB_TIP][0] < landmarks[THUMB_IP][0])

        # Index, middle, ring, pinky: Tip y-value < PIP y-value means extended.
        # (In screen coordinates, y increases downward.)
        finger_tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        finger_pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            states.append(landmarks[tip_idx][1] < landmarks[pip_idx][1])

        return states

    def _is_pinch(self, landmarks: List[Tuple[int, int]]) -> bool:
        """
        Check if the thumb tip and index finger tip are close enough for a pinch.

        Args:
            landmarks: List of 21 (x, y) pixel coordinates.

        Returns:
            True if the Euclidean distance between thumb tip and index tip
            is below the configured threshold.
        """
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        distance = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
        return distance < PINCH_DISTANCE_THRESHOLD

    def _map_finger_states_to_gesture(self, states: List[bool]) -> Gesture:
        """
        Map a finger extension pattern to the corresponding gesture.

        The thumb state is intentionally ignored for DRAW and NAVIGATE gestures
        because the thumb naturally protrudes in varying positions when the user
        points or makes a peace sign. Only the four main fingers (index, middle,
        ring, pinky) drive most gesture distinctions. The thumb is only decisive
        for REDO (thumb-only) and UNDO (fist).

        Priority order matters: REDO is checked before DRAW to prevent a
        thumb-up from being misclassified as an idle state.

        Args:
            states: [thumb, index, middle, ring, pinky] extension booleans.

        Returns:
            The gesture matching the finger pattern, or IDLE if unrecognized.
        """
        thumb, index, middle, ring, pinky = states

        # Fist: All fingers curled including thumb (undo)
        if not any(states):
            return Gesture.UNDO

        # Thumb only: Thumb extended, all others curled (redo).
        # Must be checked before DRAW to avoid being swallowed.
        if thumb and not index and not middle and not ring and not pinky:
            return Gesture.REDO

        # Index finger up, middle/ring/pinky down (thumb ignored): Draw mode.
        # This accommodates the natural thumb position while pointing.
        if index and not middle and not ring and not pinky:
            return Gesture.DRAW

        # Index + Middle up, ring/pinky down (thumb ignored): Navigate mode.
        if index and middle and not ring and not pinky:
            return Gesture.NAVIGATE

        # All fingers extended: Open palm (idle / stop)
        if all(states):
            return Gesture.IDLE

        # Any unrecognized pattern defaults to idle
        return Gesture.IDLE

    def _apply_debounce(self, raw_gesture: Gesture) -> Gesture:
        """
        Stabilize gesture output by requiring consistent detection over
        multiple consecutive frames before accepting a new gesture.

        Also enforces a cooldown period after undo/redo gestures to prevent
        repeated triggers from a single held gesture.

        Args:
            raw_gesture: The gesture detected in the current frame.

        Returns:
            The debounced gesture that should be acted upon.
        """
        # Tick down the undo/redo cooldown
        if self._undo_redo_cooldown > 0:
            self._undo_redo_cooldown -= 1

        # If the raw gesture matches the current candidate, increment counter
        if raw_gesture == self._candidate_gesture:
            self._candidate_count += 1
        else:
            # New candidate detected, reset the counter
            self._candidate_gesture = raw_gesture
            self._candidate_count = 1

        # Accept the candidate once it has persisted long enough
        if self._candidate_count >= GESTURE_DEBOUNCE_FRAMES:
            # For undo/redo, enforce cooldown to prevent rapid repeats
            if raw_gesture in (Gesture.UNDO, Gesture.REDO):
                if self._undo_redo_cooldown > 0:
                    return self._prev_gesture
                self._undo_redo_cooldown = UNDO_REDO_COOLDOWN_FRAMES

            self._prev_gesture = raw_gesture
            return raw_gesture

        # If the candidate hasn't stabilized yet, return the previous gesture
        return self._prev_gesture

    def reset(self):
        """Reset the recognizer state. Useful when restarting the application."""
        self._prev_gesture = Gesture.IDLE
        self._candidate_gesture = Gesture.IDLE
        self._candidate_count = 0
        self._undo_redo_cooldown = 0
