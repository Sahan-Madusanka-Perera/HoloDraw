"""
HoloDraw Application Controller
---------------------------------
The central coordinator that ties together hand tracking, gesture recognition,
the drawing engine, and the UI overlay into a cohesive real-time application.

This module implements the main processing loop:
    1. Capture a frame from the webcam.
    2. Detect hand landmarks.
    3. Classify the current gesture.
    4. Execute the corresponding drawing action.
    5. Composite the canvas overlay and UI onto the display frame.
    6. Handle keyboard input as backup controls.
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple
from datetime import datetime

from config import (
    Tool, Gesture, BrushSize, AppSettings,
    TOOLBAR_HEIGHT, STATUS_BAR_HEIGHT,
    CANVAS_OPACITY, MIN_DRAW_DISTANCE,
    COLOR_BLACK, COLOR_PALETTE, ERASER_RADIUS,
    UI_CURSOR_COLOR, UI_CURSOR_NAV_COLOR,
)
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from drawing_engine import DrawingEngine
from shape_tools import preview_shape
from ui_overlay import UIOverlay


class AppController:
    """
    Main application controller that runs the webcam processing loop.

    Manages all application state including the active tool, color, brush size,
    fill mode, and the shape placement workflow. Translates gestures into
    drawing actions and handles both gesture-based and keyboard-based input.

    Attributes:
        settings: Application configuration settings.
        tracker: Hand landmark detection module.
        recognizer: Gesture classification module.
        engine: Drawing canvas and history manager.
        ui: Toolbar and status bar renderer.
        active_tool: Currently selected drawing tool.
        active_color: Currently selected drawing color.
        brush_size: Current brush thickness.
        filled: Whether shapes are drawn filled or as outlines.
    """

    def __init__(self, settings: Optional[AppSettings] = None, mode = "holodraw"):
        """
        Initialize all modules and set default application state.

        Args:
            settings: Optional custom settings. Uses defaults if not provided.
        """
        self.settings = settings or AppSettings()
        self.mode = mode

        # Core modules
        self.tracker = HandTracker(
            max_hands=1,
            detection_confidence=self.settings.detection_confidence,
            tracking_confidence=self.settings.tracking_confidence,
        )
        self.recognizer = GestureRecognizer()
        self.engine: Optional[DrawingEngine] = None  # Initialized after camera opens
        self.ui: Optional[UIOverlay] = None

        # Drawing state
        self.active_tool: Tool = Tool.FREEHAND
        self.active_color: Tuple[int, int, int] = COLOR_BLACK
        self.brush_size: BrushSize = BrushSize.MEDIUM
        self.filled: bool = False

        # Cursor and drawing tracking
        self._prev_draw_point: Optional[Tuple[int, int]] = None
        self._shape_start: Optional[Tuple[int, int]] = None
        self._is_placing_shape: bool = False
        self._was_drawing: bool = False

        # Toolbar selection state (prevent repeated triggers from a held pinch)
        self._select_cooldown: int = 0
        self._SELECT_COOLDOWN_FRAMES: int = 15

        # FPS tracking
        self._fps: float = 0.0
        self._frame_time: float = time.time()

    def run(self) -> None:
        """
        Start the main application loop.

        Opens the webcam, processes frames until the user quits (press 'q'),
        and releases all resources on exit. This is the primary entry point
        called from main.py.
        """
        cap = cv2.VideoCapture(self.settings.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)

        if not cap.isOpened():
            print("[HoloDraw] Error: Could not open webcam.")
            print("[HoloDraw] Please check that a camera is connected and accessible.")
            return

        # Read actual dimensions (camera may adjust requested resolution)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize drawing engine and UI with actual frame dimensions
        self.engine = DrawingEngine(actual_width, actual_height)
        self.ui = UIOverlay(actual_width, actual_height)

        print(f"[HoloDraw] Application started. Resolution: {actual_width}x{actual_height}")
        print("[HoloDraw] Keyboard shortcuts: Q=Quit, S=Save, Z=Undo, Y=Redo, C=Clear")
        print("[HoloDraw] Use hand gestures to draw. See README for gesture guide.")

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("[HoloDraw] Warning: Failed to read frame from webcam.")
                    break

                # Mirror the frame for natural interaction
                if self.settings.mirror_mode:
                    frame = cv2.flip(frame, 1)

                # --- Hand detection ---
                results = self.tracker.detect(frame)
                hand_detected = results is not None
                gesture = Gesture.IDLE
                cursor_pos: Optional[Tuple[int, int]] = None

                if hand_detected:
                    frame_shape = (frame.shape[0], frame.shape[1])
                    lm_list = self.tracker.get_landmark_list(results, frame_shape)
                    tips = self.tracker.get_fingertip_positions(results, frame_shape)

                    # Classify the current gesture
                    gesture = self.recognizer.classify(lm_list)
                    cursor_pos = tips["index"]

                    # Optionally draw landmarks for debugging
                    if self.settings.show_landmarks:
                        self.tracker.draw_landmarks_on_frame(frame, results)

                    # Process the gesture
                    self._process_gesture(gesture, cursor_pos)
                else:
                    # No hand detected: finalize any ongoing stroke
                    self._finalize_drawing_state()

                # --- Compose display frame ---
                display = self._compose_display(frame, cursor_pos, gesture)

                # --- Render UI ---
                self.ui.render(
                    display,
                    active_tool=self.active_tool,
                    active_color=self.active_color,
                    brush_size=self.brush_size,
                    filled=self.filled,
                    fps=self._fps,
                    hand_detected=hand_detected,
                    gesture_name=gesture.name if hand_detected else "",
                )

                # --- Show the frame ---
                cv2.imshow(self.settings.window_name, display)

                # --- Update FPS ---
                self._update_fps()

                # --- Handle keyboard input ---
                if self._handle_keyboard():
                    break

        finally:
            cap.release()
            self.tracker.release()
            cv2.destroyAllWindows()
            print("[HoloDraw] Application closed.")

    def _process_gesture(
        self,
        gesture: Gesture,
        cursor_pos: Tuple[int, int],
    ) -> None:
        """
        Execute the action corresponding to the detected gesture.

        Args:
            gesture: The classified gesture for the current frame.
            cursor_pos: (x, y) pixel position of the index fingertip.
        """
        # Tick down the select cooldown
        if self._select_cooldown > 0:
            self._select_cooldown -= 1

        if gesture == Gesture.DRAW:
            self._handle_draw(cursor_pos)

        elif gesture == Gesture.NAVIGATE:
            self._handle_navigate(cursor_pos)

        elif gesture == Gesture.SELECT:
            self._handle_select(cursor_pos)

        elif gesture == Gesture.UNDO:
            self._handle_undo()

        elif gesture == Gesture.REDO:
            self._handle_redo()

        elif gesture == Gesture.IDLE:
            self._finalize_drawing_state()

    def _handle_draw(self, pos: Tuple[int, int]) -> None:
        """
        Handle the DRAW gesture for the current frame.

        For freehand and eraser tools, draws continuously between frames.
        For shape tools, records the start point and waits for the gesture
        to end (transition to NAVIGATE) to finalize the shape.

        Args:
            pos: Current cursor position (index fingertip).
        """
        # Ignore drawing in the toolbar or status bar regions
        if pos[1] < TOOLBAR_HEIGHT or pos[1] > (self.engine.height - STATUS_BAR_HEIGHT):
            return

        if self.active_tool == Tool.FREEHAND:
            # Begin a new stroke if not already drawing
            if not self._was_drawing:
                self.engine.begin_stroke()
                self._was_drawing = True
                self._prev_draw_point = pos
                return

            # Draw segment if the cursor moved enough (reduces jitter)
            if self._prev_draw_point is not None:
                dist = np.hypot(
                    pos[0] - self._prev_draw_point[0],
                    pos[1] - self._prev_draw_point[1],
                )
                if dist >= MIN_DRAW_DISTANCE:
                    self.engine.draw_freehand(
                        self._prev_draw_point, pos,
                        self.active_color, self.brush_size.value,
                    )
                    self._prev_draw_point = pos

        elif self.active_tool == Tool.ERASER:
            if not self._was_drawing:
                self.engine.begin_erase()
                self._was_drawing = True
            self.engine.erase(pos, ERASER_RADIUS)

        elif self.active_tool in (Tool.LINE, Tool.RECTANGLE, Tool.CIRCLE, Tool.ELLIPSE):
            # Shape tools: record start point on first draw frame
            if not self._is_placing_shape:
                self._shape_start = pos
                self._is_placing_shape = True
                self._was_drawing = True

    def _handle_navigate(self, pos: Tuple[int, int]) -> None:
        """
        Handle the NAVIGATE gesture.

        If we were previously drawing a shape, this finalizes the shape
        placement. Otherwise, it simply resets the drawing state.

        Args:
            pos: Current cursor position.
        """
        if self._is_placing_shape and self._shape_start is not None:
            # Commit the shape to the canvas
            self.engine.commit_shape(
                self.active_tool,
                self._shape_start, pos,
                self.active_color, self.brush_size.value,
                self.filled,
            )
            self._is_placing_shape = False
            self._shape_start = None
            self._was_drawing = False

        elif self._was_drawing:
            # End a freehand stroke or erase sequence
            self.engine.end_stroke()
            self._was_drawing = False
            self._prev_draw_point = None

    def _handle_select(self, pos: Tuple[int, int]) -> None:
        """
        Handle the SELECT (pinch) gesture for toolbar interaction.

        Performs a hit-test against toolbar items and updates the
        corresponding application state.

        Args:
            pos: Current cursor position (pinch point).
        """
        # Prevent rapid repeated selections from a held pinch
        if self._select_cooldown > 0:
            return

        # Finalize any ongoing drawing
        self._finalize_drawing_state()

        # Only test toolbar region
        if pos[1] > TOOLBAR_HEIGHT:
            return

        item = self.ui.hit_test(pos)
        if item is None:
            return

        self._select_cooldown = self._SELECT_COOLDOWN_FRAMES

        if item.action == "tool":
            self.active_tool = item.value

        elif item.action == "color":
            self.active_color = item.value

        elif item.action == "brush":
            self.brush_size = item.value

        elif item.action == "fill":
            self.filled = not self.filled

        elif item.action == "clear":
            self.engine.clear()

        elif item.action == "save":
            self._save_canvas()

    def _handle_undo(self) -> None:
        """Execute an undo operation."""
        self._finalize_drawing_state()
        self.engine.undo()

    def _handle_redo(self) -> None:
        """Execute a redo operation."""
        self._finalize_drawing_state()
        self.engine.redo()

    def _finalize_drawing_state(self) -> None:
        """
        Clean up any in-progress drawing state.

        Called when the hand is lost, the gesture changes to IDLE,
        or before performing undo/redo/selection.
        """
        if self._was_drawing:
            self.engine.end_stroke()
            self._was_drawing = False
            self._prev_draw_point = None

        if self._is_placing_shape:
            # Cancel shape placement if interrupted
            self._is_placing_shape = False
            self._shape_start = None

    def _compose_display(
        self,
        frame: np.ndarray,
        cursor_pos: Optional[Tuple[int, int]],
        gesture: Gesture,
    ) -> np.ndarray:
        """
        Composite the camera frame, drawing canvas, and cursor into a
        single display image.

        The canvas is overlaid onto the camera feed with configurable opacity.
        Only non-white pixels from the canvas are blended to keep the camera
        feed visible through empty canvas areas.

        Args:
            frame: The raw camera frame.
            cursor_pos: Current cursor position, or None if no hand detected.
            gesture: Current gesture (affects cursor appearance).

        Returns:
            The composited display frame.
        """
        canvas = self.engine.get_canvas()
        display = frame.copy()

        # Create a mask of drawn pixels (non-white areas on the canvas)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Blend drawn areas onto the camera feed with high opacity
        # so the drawing is clearly visible while the camera feed shows through
        # undrawn regions.
        draw_opacity = min(CANVAS_OPACITY + 0.4, 1.0)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        display = (
            display * (1 - mask_3ch * draw_opacity)
            + canvas * (mask_3ch * draw_opacity)
        ).astype(np.uint8)

        # Draw shape preview if currently placing a shape
        if self._is_placing_shape and self._shape_start and cursor_pos:
            display = preview_shape(
                display, self.active_tool,
                self._shape_start, cursor_pos,
                self.active_color, 2,
            )

        # Draw cursor
        if cursor_pos is not None:
            self._draw_cursor(display, cursor_pos, gesture)

        return display

    def _draw_cursor(
        self,
        frame: np.ndarray,
        pos: Tuple[int, int],
        gesture: Gesture,
    ) -> None:
        """
        Draw a visual cursor at the index fingertip position.

        The cursor appearance changes based on the current gesture and tool:
        - DRAW mode: Solid circle (size matches brush or eraser)
        - NAVIGATE mode: Crosshair
        - SELECT mode: Small dot
        - Other: Simple ring

        Args:
            frame: Display frame to draw on.
            pos: (x, y) cursor position.
            gesture: Current gesture for styling the cursor.
        """
        if gesture == Gesture.DRAW:
            if self.active_tool == Tool.ERASER:
                # Show eraser circle outline
                cv2.circle(frame, pos, ERASER_RADIUS, (180, 180, 180), 2, cv2.LINE_AA)
            else:
                # Show brush cursor
                radius = max(self.brush_size.value // 2, 3)
                cv2.circle(frame, pos, radius, UI_CURSOR_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, pos, radius + 2, (0, 200, 100), 1, cv2.LINE_AA)

        elif gesture == Gesture.NAVIGATE:
            # Crosshair cursor
            size = 12
            cv2.line(frame, (pos[0] - size, pos[1]), (pos[0] + size, pos[1]),
                     UI_CURSOR_NAV_COLOR, 1, cv2.LINE_AA)
            cv2.line(frame, (pos[0], pos[1] - size), (pos[0], pos[1] + size),
                     UI_CURSOR_NAV_COLOR, 1, cv2.LINE_AA)
            cv2.circle(frame, pos, 3, UI_CURSOR_NAV_COLOR, -1, cv2.LINE_AA)

        elif gesture == Gesture.SELECT:
            # Small selection dot
            cv2.circle(frame, pos, 6, UI_CURSOR_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, pos, 10, UI_CURSOR_COLOR, 1, cv2.LINE_AA)

        else:
            # Default cursor ring
            cv2.circle(frame, pos, 8, UI_CURSOR_COLOR, 2, cv2.LINE_AA)

    def _save_canvas(self) -> None:
        """
        Save the current canvas to a PNG file with a timestamped filename.

        Files are saved to the current working directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"holodraw_{timestamp}.png"
        if self.engine.save(filepath):
            print(f"[HoloDraw] Drawing saved to: {filepath}")
        else:
            print(f"[HoloDraw] Error: Failed to save drawing to {filepath}")

    def _update_fps(self) -> None:
        """Calculate and update the rolling frames-per-second value."""
        current_time = time.time()
        elapsed = current_time - self._frame_time
        if elapsed > 0:
            # Exponential moving average for smooth FPS display
            instant_fps = 1.0 / elapsed
            self._fps = self._fps * 0.9 + instant_fps * 0.1
        self._frame_time = current_time

    def _handle_keyboard(self) -> bool:
        """
        Process keyboard input for backup controls.

        Returns:
            True if the application should quit, False otherwise.
        """
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            return True

        elif key == ord('s') or key == ord('S'):
            self._save_canvas()

        elif key == ord('z') or key == ord('Z'):
            self.engine.undo()

        elif key == ord('y') or key == ord('Y'):
            self.engine.redo()

        elif key == ord('c') or key == ord('C'):
            self.engine.clear()

        elif key == ord('f') or key == ord('F'):
            self.filled = not self.filled

        elif key == ord('l') or key == ord('L'):
            # Toggle landmark visibility for debugging
            self.settings.show_landmarks = not self.settings.show_landmarks

        return False
