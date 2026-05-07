"""
HoloPaint Controller Module
-------------------------------
Dedicated controller for the HoloPaint mode. This mode allows users to:
    1. Fetch a random image from the internet or upload a local image.
    2. Extract edges/outlines using a trained U-Net deep learning model.
    3. Paint over the outlines using hand gestures (same gesture system as HoloDraw).

The controller manages a two-layer canvas system:
    - Base layer: The edge-detected outline image (non-erasable).
    - Paint layer: User's painted strokes on a transparent overlay.

The final display composites: Camera feed → Base outlines → User paint → UI.
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple
from datetime import datetime

from config import (
    Tool, Gesture, BrushSize, AppSettings,
    TOOLBAR_HEIGHT, STATUS_BAR_HEIGHT,
    MIN_DRAW_DISTANCE,
    COLOR_BLACK, COLOR_WHITE, COLOR_PALETTE, ERASER_RADIUS,
    UI_CURSOR_COLOR, UI_CURSOR_NAV_COLOR,
    HOLOPAINT_DEFAULT_THRESHOLD, HOLOPAINT_THRESHOLD_STEP,
    EDGE_MODEL_PATH,
)
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from drawing_engine import DrawingEngine
from ui_overlay import UIOverlay
from edge_detector import EdgeDetector
from image_fetcher import ImageFetcher


class HoloPaintController:
    """
    Controller for the HoloPaint mode: image → edge detection → paint-over.

    Manages the complete workflow from image acquisition through edge detection
    to gesture-based painting. Reuses the existing hand tracking and gesture
    recognition infrastructure from HoloDraw.

    Attributes:
        settings: Application configuration.
        tracker: Hand landmark detection module.
        recognizer: Gesture classification module.
        engine: Drawing canvas for user's paint strokes.
        ui: Toolbar and status bar renderer.
        edge_detector: U-Net-based edge detection module.
        image_fetcher: Random image acquisition module.
        base_outline: The current edge-detected outline (base layer).
        source_image: The original source image before edge detection.
        threshold: Current edge detection sensitivity threshold.
    """

    # States for the HoloPaint workflow
    STATE_IMAGE_SELECT = "image_select"
    STATE_THRESHOLD_ADJUST = "threshold_adjust"
    STATE_PAINTING = "painting"

    def __init__(self, settings: Optional[AppSettings] = None):
        """
        Initialize the HoloPaint controller.

        Args:
            settings: Optional custom settings. Uses defaults if not provided.
        """
        self.settings = settings or AppSettings()

        # Core modules (shared with HoloDraw)
        self.tracker = HandTracker(
            max_hands=1,
            detection_confidence=self.settings.detection_confidence,
            tracking_confidence=self.settings.tracking_confidence,
        )
        self.recognizer = GestureRecognizer()
        self.engine: Optional[DrawingEngine] = None
        self.ui: Optional[UIOverlay] = None

        # HoloPaint-specific modules
        self.edge_detector = EdgeDetector(model_path=EDGE_MODEL_PATH)
        self.image_fetcher = ImageFetcher()

        # Image state
        self.source_image: Optional[np.ndarray] = None
        self.base_outline: Optional[np.ndarray] = None
        self._fitted_outline: Optional[np.ndarray] = None  # Pre-computed letterboxed outline
        self._outline_offset: Tuple[int, int] = (0, 0)     # (x, y) offset of fitted image
        self._outline_fitted_size: Tuple[int, int] = (0, 0) # (w, h) of fitted image
        self.threshold: float = HOLOPAINT_DEFAULT_THRESHOLD

        # Drawing state
        self.active_color: Tuple[int, int, int] = COLOR_BLACK
        self.brush_size: BrushSize = BrushSize.MEDIUM
        self._prev_draw_point: Optional[Tuple[int, int]] = None
        self._was_drawing: bool = False

        # Fill mode state
        self._fill_mode: bool = False
        self._fill_cooldown: int = 0
        self._FILL_COOLDOWN_FRAMES: int = 10
        self._last_cursor_pos: Optional[Tuple[int, int]] = None  # For keyboard fill

        # Cursor smoothing state
        self._smoothed_cursor_pos: Optional[np.ndarray] = None
        self._CURSOR_SMOOTHING_ALPHA: float = 0.35  # Lower = smoother, Higher = more responsive

        # Mouse input state
        self._mouse_down: bool = False

        # Toolbar selection cooldown
        self._select_cooldown: int = 0
        self._SELECT_COOLDOWN_FRAMES: int = 15

        # Zoom and pan state
        self._zoom: float = 1.0
        self._zoom_min: float = 1.0
        self._zoom_max: float = 5.0
        self._zoom_step: float = 0.25
        self._pan_x: float = 0.0   # Pan offset in normalized coords (0.0 = centered)
        self._pan_y: float = 0.0
        self._pan_step: int = 30    # Pixels to pan per key press

        # Gesture-based zoom tracking
        self._zoom_gesture_active: bool = False     # True while three-finger gesture is held
        self._zoom_anchor: Optional[Tuple[int, int]] = None  # Hand position when zoom gesture started
        self._zoom_dead_zone: int = 25              # Pixels of movement before zoom/pan activates
        self._zoom_sensitivity: float = 0.008       # Zoom change per pixel of vertical movement
        self._pan_sensitivity: float = 1.5          # Pan pixels per pixel of horizontal movement
        self._zoom_direction: str = ""               # Current zoom direction for visual feedback

        # Application state
        self._state: str = self.STATE_IMAGE_SELECT
        self._fps: float = 0.0
        self._frame_time: float = time.time()

    def run(self) -> None:
        """
        Start the HoloPaint application loop.

        Shows the image selection screen first, then transitions to the
        painting canvas once an image has been processed.
        """
        cap = cv2.VideoCapture(self.settings.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)

        if not cap.isOpened():
            print("[HoloPaint] Error: Could not open webcam.")
            return

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize drawing engine and UI
        self.engine = DrawingEngine(actual_width, actual_height)
        self.ui = UIOverlay(actual_width, actual_height)

        cv2.namedWindow("HoloPaint - Paint Over Outlines")
        cv2.setMouseCallback("HoloPaint - Paint Over Outlines", self._mouse_callback)

        print(f"[HoloPaint] Application started. Resolution: {actual_width}x{actual_height}")
        print("[HoloPaint] Press R to fetch a random image, U to upload, Q to quit.")

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("[HoloPaint] Warning: Failed to read frame.")
                    break

                if self.settings.mirror_mode:
                    frame = cv2.flip(frame, 1)

                # Route to the appropriate state handler
                if self._state == self.STATE_IMAGE_SELECT:
                    display = self._render_image_select_screen(frame)
                elif self._state == self.STATE_THRESHOLD_ADJUST:
                    display = self._render_threshold_screen(frame)
                elif self._state == self.STATE_PAINTING:
                    display = self._run_painting_loop(frame)
                else:
                    display = frame

                cv2.imshow("HoloPaint - Paint Over Outlines", display)
                self._update_fps()

                if self._handle_keyboard():
                    break

        finally:
            cap.release()
            self.tracker.release()
            cv2.destroyAllWindows()
            print("[HoloPaint] Application closed.")

    # -----------------------------------------------------------------------
    # State: Image Selection
    # -----------------------------------------------------------------------

    def _render_image_select_screen(self, frame: np.ndarray) -> np.ndarray:
        """Render the image source selection overlay."""
        display = frame.copy()

        # Semi-transparent dark overlay
        overlay = np.zeros_like(display)
        cv2.addWeighted(display, 0.3, overlay, 0.7, 0, display)

        h, w = display.shape[:2]

        # Title
        cv2.putText(
            display, "HoloPaint", (w // 2 - 160, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 230, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            display, "Choose an image source:", (w // 2 - 180, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA,
        )

        # Option cards
        card_w, card_h = 400, 180

        # Card 1: Random Image
        cx1 = w // 2 - card_w // 2
        cy1 = 220
        self._draw_option_card(
            display, cx1, cy1, card_w, card_h,
            "R", "Random Image",
            "Fetch a random photo from the internet",
            (0, 200, 200),
        )

        # Card 2: Upload Image
        cx2 = w // 2 - card_w // 2
        cy2 = 420
        self._draw_option_card(
            display, cx2, cy2, card_w, card_h,
            "U", "Upload Image",
            "Load an image from your computer",
            (200, 150, 50),
        )

        # Footer
        cv2.putText(
            display, "Press R or U to select, Q to quit",
            (w // 2 - 180, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA,
        )

        return display

    def _draw_option_card(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        shortcut: str, title: str, subtitle: str,
        accent_color: Tuple[int, int, int],
    ) -> None:
        """Draw a selection card on the screen."""
        # Card background
        self.ui._draw_rounded_rect(frame, (x, y), (x + w, y + h), (32, 28, 24), -1, 15)
        self.ui._draw_rounded_rect(frame, (x, y), (x + w, y + h), (70, 60, 55), 2, 15)

        # Sleek Accent stripe on the left
        cv2.line(frame, (x + 12, y + 25), (x + 12, y + h - 25), accent_color, 4, cv2.LINE_AA)

        # Shortcut key
        cv2.putText(
            frame, f"[{shortcut}]", (x + 30, y + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, accent_color, 3, cv2.LINE_AA,
        )

        # Title
        cv2.putText(
            frame, title, (x + 110, y + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Subtitle
        cv2.putText(
            frame, subtitle, (x + 30, y + 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # State: Threshold Adjustment
    # -----------------------------------------------------------------------

    def _render_threshold_screen(self, frame: np.ndarray) -> np.ndarray:
        """Render the threshold adjustment screen with live preview."""
        if self.source_image is None:
            self._state = self.STATE_IMAGE_SELECT
            return frame

        h, w = frame.shape[:2]
        display = np.zeros((h, w, 3), dtype=np.uint8)
        display[:] = (25, 25, 30)

        # Show source image (left half) and edge preview (right half)
        preview_w = w // 2 - 20
        preview_h = h - 160

        # Left: Source image
        src_resized = cv2.resize(self.source_image, (preview_w, preview_h))
        display[100:100 + preview_h, 10:10 + preview_w] = src_resized

        # Right: Edge detection preview
        if self.base_outline is not None:
            edge_resized = cv2.resize(self.base_outline, (preview_w, preview_h))
            display[100:100 + preview_h, w // 2 + 10:w // 2 + 10 + preview_w] = edge_resized

        # Title
        cv2.putText(
            display, "Adjust Edge Threshold", (w // 2 - 180, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
        )

        # Labels
        cv2.putText(
            display, "Original", (10 + preview_w // 2 - 40, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
        )
        cv2.putText(
            display, "Edge Outline", (w // 2 + 10 + preview_w // 2 - 60, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Threshold bar
        bar_y = h - 50
        bar_x1 = 100
        bar_x2 = w - 100
        bar_w = bar_x2 - bar_x1

        # Background bar
        cv2.rectangle(display, (bar_x1, bar_y - 8), (bar_x2, bar_y + 8), (60, 60, 60), -1)

        # Filled portion
        fill_x = int(bar_x1 + bar_w * self.threshold)
        cv2.rectangle(display, (bar_x1, bar_y - 8), (fill_x, bar_y + 8), (0, 200, 200), -1)

        # Thumb
        cv2.circle(display, (fill_x, bar_y), 12, (0, 255, 255), -1, cv2.LINE_AA)

        # Threshold label
        cv2.putText(
            display, f"Threshold: {self.threshold:.2f}", (w // 2 - 80, bar_y - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Instructions
        cv2.putText(
            display, "[+/-] Adjust  |  [Enter] Confirm  |  [N] New Image  |  [Q] Quit",
            (w // 2 - 300, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
        )

        return display

    def _process_current_image(self) -> None:
        """Run edge detection on the current source image."""
        if self.source_image is not None:
            print(f"[HoloPaint] Running edge detection (threshold={self.threshold:.2f})...")
            self.base_outline = self.edge_detector.detect_edges(
                self.source_image, threshold=self.threshold,
            )
            print("[HoloPaint] Edge detection complete.")

    # -----------------------------------------------------------------------
    # State: Painting
    # -----------------------------------------------------------------------

    def _run_painting_loop(self, frame: np.ndarray) -> np.ndarray:
        """Run one frame of the painting mode with gesture-based drawing."""
        # --- Hand detection ---
        results = self.tracker.detect(frame)
        hand_detected = results is not None
        gesture = Gesture.IDLE
        cursor_pos: Optional[Tuple[int, int]] = None

        # Tick fill cooldown
        if self._fill_cooldown > 0:
            self._fill_cooldown -= 1

        if hand_detected:
            frame_shape = (frame.shape[0], frame.shape[1])
            lm_list = self.tracker.get_landmark_list(results, frame_shape)
            tips = self.tracker.get_fingertip_positions(results, frame_shape)

            gesture = self.recognizer.classify(lm_list)
            raw_cursor_pos = tips["index"]

            # --- Cursor Smoothing (EMA) ---
            if self._smoothed_cursor_pos is None:
                self._smoothed_cursor_pos = np.array(raw_cursor_pos, dtype=float)
            else:
                self._smoothed_cursor_pos = (
                    self._CURSOR_SMOOTHING_ALPHA * np.array(raw_cursor_pos)
                    + (1.0 - self._CURSOR_SMOOTHING_ALPHA) * self._smoothed_cursor_pos
                )
            
            cursor_pos = (int(self._smoothed_cursor_pos[0]), int(self._smoothed_cursor_pos[1]))

            if self.settings.show_landmarks:
                self.tracker.draw_landmarks_on_frame(frame, results)

            # Ignore hand gestures if mouse is currently being used to paint
            if not self._mouse_down:
                self._process_gesture(gesture, cursor_pos)
        else:
            self._smoothed_cursor_pos = None
            if not self._mouse_down:
                self._finalize_drawing_state()

        # --- Compose display ---
        display = self._compose_paint_display(frame, cursor_pos, gesture)

        # --- Render simplified toolbar ---
        self.ui.render(
            display,
            active_tool=Tool.FREEHAND,
            active_color=self.active_color,
            brush_size=self.brush_size,
            filled=self._fill_mode,
            fps=self._fps,
            hand_detected=hand_detected,
            gesture_name=gesture.name if hand_detected else "",
        )

        h, w = display.shape[:2]

        # HoloPaint mode indicator
        if self._zoom_gesture_active:
            mode_label = "ZOOM MODE"
            mode_color = (0, 220, 220)  # Cyan for zoom
        elif self._fill_mode:
            mode_label = "FILL MODE"
            mode_color = (0, 200, 0)
        else:
            mode_label = "HOLOPAINT"
            mode_color = (0, 255, 255)

        cv2.putText(
            display, mode_label, (w - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA,
        )

        # Zoom level indicator
        if self._zoom > 1.01:
            zoom_text = f"Zoom: {self._zoom:.1f}x"
            cv2.putText(
                display, zoom_text, (w - 180, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA,
            )
            self._draw_minimap(display)

        # Zoom gesture hint (show briefly when not zoomed)
        if self._zoom_gesture_active and self._zoom <= 1.01:
            cv2.putText(
                display, "Move hand UP to zoom in", (w // 2 - 130, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA,
            )

        # --- Camera PiP preview (so users can see their hand) ---
        self._draw_camera_pip(display, frame)

        return display

    def _draw_camera_pip(
        self,
        display: np.ndarray,
        camera_frame: np.ndarray,
    ) -> None:
        """
        Draw a small camera picture-in-picture preview in the bottom-left
        corner so the user can see their hand position while the main canvas
        shows the white outline page.
        """
        h, w = display.shape[:2]

        # PiP dimensions (16:9 aspect, fitting in bottom-left)
        pip_w = 200
        pip_h = int(pip_w * camera_frame.shape[0] / camera_frame.shape[1])
        pip_x = 10
        pip_y = h - pip_h - 45  # Above status bar

        # Resize camera frame to PiP size
        pip_frame = cv2.resize(camera_frame, (pip_w, pip_h))

        # Semi-transparent border
        cv2.rectangle(
            display,
            (pip_x - 2, pip_y - 2),
            (pip_x + pip_w + 2, pip_y + pip_h + 2),
            (60, 60, 60), 2,
        )

        # Place PiP
        display[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = pip_frame

        # Label
        cv2.putText(
            display, "Camera", (pip_x + 5, pip_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
        )

    def _fit_outline_to_canvas(self) -> None:
        """
        Pre-compute the outline image fitted to the canvas dimensions
        with correct aspect ratio (letterboxed on white background).

        This is called once when entering painting mode so the fitted
        image doesn't need to be recomputed every frame.
        """
        if self.base_outline is None or self.engine is None:
            return

        canvas_w, canvas_h = self.engine.width, self.engine.height
        src_h, src_w = self.base_outline.shape[:2]

        # Calculate scale to fit within canvas while preserving aspect ratio
        scale = min(canvas_w / src_w, canvas_h / src_h)
        fitted_w = int(src_w * scale)
        fitted_h = int(src_h * scale)

        # Center offset (letterbox padding)
        offset_x = (canvas_w - fitted_w) // 2
        offset_y = (canvas_h - fitted_h) // 2

        # Create white canvas and place the fitted outline in the center
        fitted = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        resized_outline = cv2.resize(
            self.base_outline, (fitted_w, fitted_h),
            interpolation=cv2.INTER_AREA,
        )
        fitted[offset_y:offset_y + fitted_h, offset_x:offset_x + fitted_w] = resized_outline

        self._fitted_outline = fitted
        self._outline_offset = (offset_x, offset_y)
        self._outline_fitted_size = (fitted_w, fitted_h)

    def _compose_paint_display(
        self,
        frame: np.ndarray,
        cursor_pos: Optional[Tuple[int, int]],
        gesture: Gesture,
    ) -> np.ndarray:
        """
        Composite a clean painting canvas with outline and user paint.

        Uses a white background (coloring-book style) instead of the
        camera feed, with the outline image letterbox-fitted to preserve
        the correct aspect ratio.

        Compositing order:
            1. White background with letterboxed outline
            2. User's paint strokes
            3. Zoom/pan viewport crop
            4. Cursor overlay

        The camera frame is still used for hand tracking but is NOT
        displayed as the background — only the outline canvas is shown.
        """
        h, w = frame.shape[:2]

        # Layer 1: Start with the pre-computed fitted outline on white
        if self._fitted_outline is not None:
            display = self._fitted_outline.copy()
        else:
            display = np.full((h, w, 3), 255, dtype=np.uint8)

        # Layer 2: Overlay user's paint strokes
        if self.engine is not None:
            canvas = self.engine.get_canvas()
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, paint_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

            if np.any(paint_mask > 0):
                paint_mask_3ch = cv2.cvtColor(paint_mask, cv2.COLOR_GRAY2BGR) / 255.0
                display = (
                    display * (1 - paint_mask_3ch)
                    + canvas * paint_mask_3ch
                ).astype(np.uint8)

        # --- Apply zoom/pan ---
        if self._zoom > 1.01:
            display = self._apply_zoom(display)

        # Draw cursor (after zoom so it's always sharp)
        if cursor_pos is not None:
            self._draw_cursor(display, cursor_pos, gesture)

        return display

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop and upscale a viewport region of the frame based on zoom/pan state.

        Args:
            frame: Full-resolution composited frame.

        Returns:
            Zoomed frame at the same output dimensions.
        """
        h, w = frame.shape[:2]

        # Calculate the viewport size at current zoom
        view_w = int(w / self._zoom)
        view_h = int(h / self._zoom)

        # Calculate viewport center with pan offset, clamped to image bounds
        cx = w // 2 + int(self._pan_x)
        cy = h // 2 + int(self._pan_y)

        # Clamp so viewport stays within frame
        x1 = max(0, min(cx - view_w // 2, w - view_w))
        y1 = max(0, min(cy - view_h // 2, h - view_h))
        x2 = x1 + view_w
        y2 = y1 + view_h

        # Store viewport bounds for cursor mapping
        self._viewport = (x1, y1, x2, y2)

        # Crop and resize back to full frame dimensions
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return zoomed

    def _screen_to_canvas(self, screen_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Map a screen-space cursor position to canvas-space coordinates,
        accounting for the current zoom and pan.

        Args:
            screen_pos: (x, y) position in the display window.

        Returns:
            (x, y) position on the actual canvas.
        """
        if self._zoom <= 1.01:
            return screen_pos

        sx, sy = screen_pos
        h, w = self.engine.height, self.engine.width

        # Map screen coordinates to viewport coordinates
        view_w = int(w / self._zoom)
        view_h = int(h / self._zoom)

        cx = w // 2 + int(self._pan_x)
        cy = h // 2 + int(self._pan_y)

        x1 = max(0, min(cx - view_w // 2, w - view_w))
        y1 = max(0, min(cy - view_h // 2, h - view_h))

        # Screen pos → canvas pos
        canvas_x = int(x1 + sx * view_w / w)
        canvas_y = int(y1 + sy * view_h / h)

        return (canvas_x, canvas_y)

    def _draw_minimap(self, display: np.ndarray) -> None:
        """
        Draw a small minimap in the bottom-right corner showing the current
        viewport position within the full image.
        """
        h, w = display.shape[:2]

        # Minimap dimensions
        mm_w, mm_h = 120, 68
        mm_x = w - mm_w - 10
        mm_y = h - mm_h - 50  # Above status bar

        # Minimap background
        cv2.rectangle(display, (mm_x, mm_y), (mm_x + mm_w, mm_y + mm_h), (30, 30, 30), -1)
        cv2.rectangle(display, (mm_x, mm_y), (mm_x + mm_w, mm_y + mm_h), (80, 80, 80), 1)

        # Viewport rectangle
        vp_w = int(mm_w / self._zoom)
        vp_h = int(mm_h / self._zoom)
        vp_cx = mm_w // 2 + int(self._pan_x * mm_w / w)
        vp_cy = mm_h // 2 + int(self._pan_y * mm_h / h)
        vp_x1 = max(0, min(vp_cx - vp_w // 2, mm_w - vp_w))
        vp_y1 = max(0, min(vp_cy - vp_h // 2, mm_h - vp_h))

        cv2.rectangle(
            display,
            (mm_x + vp_x1, mm_y + vp_y1),
            (mm_x + vp_x1 + vp_w, mm_y + vp_y1 + vp_h),
            (0, 255, 255), 1,
        )

    # -----------------------------------------------------------------------
    # Gesture Processing (simplified for paint mode)
    # -----------------------------------------------------------------------

    def _process_gesture(
        self,
        gesture: Gesture,
        cursor_pos: Tuple[int, int],
    ) -> None:
        """Process gestures for paint mode (freehand, fill, or toolbar select)."""
        if self._select_cooldown > 0:
            self._select_cooldown -= 1

        # Always track cursor position for keyboard-triggered fill
        self._last_cursor_pos = cursor_pos

        # --- Zoom mode gesture (three-finger: index+middle+ring) ---
        # This is checked first and handled independently of fill/paint mode.
        if gesture == Gesture.ZOOM_MODE:
            self._finalize_drawing_state()
            self._handle_zoom_gesture(cursor_pos)
            return
        else:
            # Exiting zoom mode: reset anchor
            if self._zoom_gesture_active:
                self._zoom_gesture_active = False
                self._zoom_anchor = None
                self._zoom_direction = ""

        if self._fill_mode:
            # --- Fill mode ---
            # All gestures just move the cursor, EXCEPT:
            # - SELECT (pinch) fills at cursor (or selects toolbar)
            # - UNDO/REDO still work
            if gesture == Gesture.SELECT:
                if cursor_pos[1] <= TOOLBAR_HEIGHT:
                    self._handle_select(cursor_pos)
                else:
                    self._handle_fill(cursor_pos)
            elif gesture == Gesture.UNDO:
                self.engine.undo()
            elif gesture == Gesture.REDO:
                self.engine.redo()
            # DRAW, NAVIGATE, IDLE all just track cursor — no action needed
        else:
            # --- Paint mode ---
            if gesture == Gesture.DRAW:
                self._handle_draw(cursor_pos)
            elif gesture == Gesture.NAVIGATE:
                self._finalize_drawing_state()
            elif gesture == Gesture.SELECT:
                self._handle_select(cursor_pos)
            elif gesture == Gesture.UNDO:
                self._finalize_drawing_state()
                self.engine.undo()
            elif gesture == Gesture.REDO:
                self._finalize_drawing_state()
                self.engine.redo()
            elif gesture == Gesture.IDLE:
                self._finalize_drawing_state()

    def _handle_fill(self, pos: Tuple[int, int]) -> None:
        """
        Flood-fill a region on the canvas at the cursor position.

        Uses the composite image (outline + existing paint) as the boundary
        reference so the fill respects both the original outlines and any
        previously painted regions.

        Args:
            pos: Screen-space cursor position.
        """
        if self._fill_cooldown > 0:
            return

        if pos[1] < TOOLBAR_HEIGHT or pos[1] > (self.engine.height - STATUS_BAR_HEIGHT):
            return

        # Map screen position to canvas position
        canvas_pos = self._screen_to_canvas(pos)
        cx, cy = canvas_pos

        h, w = self.engine.height, self.engine.width
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return

        # Build the reference image: outline + existing paint
        if self._fitted_outline is not None:
            ref = self._fitted_outline.copy()
        else:
            ref = np.full((h, w, 3), 255, dtype=np.uint8)

        # Overlay existing paint onto reference
        canvas = self.engine.get_canvas()
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, paint_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        if np.any(paint_mask > 0):
            pm3 = cv2.cvtColor(paint_mask, cv2.COLOR_GRAY2BGR) / 255.0
            ref = (ref * (1 - pm3) + canvas * pm3).astype(np.uint8)

        # Convert to grayscale for flood fill boundary detection
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

        # Create mask for floodFill (must be 2px larger than image)
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill on the reference to find the region
        # lo/hi difference: how tolerant the fill is to color variation
        lo_diff = (25, 25, 25)
        hi_diff = (25, 25, 25)

        # Use a copy for the fill detection
        ref_copy = ref.copy()
        _, _, filled_mask, _ = cv2.floodFill(
            ref_copy, ff_mask, (cx, cy), (0, 0, 0),
            loDiff=lo_diff, upDiff=hi_diff,
            flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
        )

        # Extract the filled region from the mask (trim the 1px border)
        region_mask = filled_mask[1:-1, 1:-1]

        if np.any(region_mask > 0):
            # Apply the fill color to the engine canvas
            self.engine.begin_stroke()
            color_bgr = self.active_color
            canvas = self.engine.get_canvas()
            canvas[region_mask > 0] = color_bgr
            self.engine.end_stroke()
            self._fill_cooldown = self._FILL_COOLDOWN_FRAMES

    def _handle_draw(self, pos: Tuple[int, int]) -> None:
        """Handle freehand drawing in paint mode, mapping through zoom."""
        if pos[1] < TOOLBAR_HEIGHT or pos[1] > (self.engine.height - STATUS_BAR_HEIGHT):
            return

        # Map screen position to canvas position (accounts for zoom/pan)
        canvas_pos = self._screen_to_canvas(pos)

        if not self._was_drawing:
            self.engine.begin_stroke()
            self._was_drawing = True
            self._prev_draw_point = canvas_pos
            return

        if self._prev_draw_point is not None:
            dist = np.hypot(
                canvas_pos[0] - self._prev_draw_point[0],
                canvas_pos[1] - self._prev_draw_point[1],
            )
            # Scale min draw distance by zoom for smoother strokes when zoomed
            min_dist = max(1, MIN_DRAW_DISTANCE / self._zoom)
            if dist >= min_dist:
                self.engine.draw_freehand(
                    self._prev_draw_point, canvas_pos,
                    self.active_color, self.brush_size.value,
                )
                self._prev_draw_point = canvas_pos

    def _handle_select(self, pos: Tuple[int, int]) -> None:
        """Handle toolbar selection or fill-mode pinch-to-fill."""
        if self._select_cooldown > 0:
            return

        self._finalize_drawing_state()

        # Only do toolbar selection if cursor is on the toolbar
        if pos[1] > TOOLBAR_HEIGHT:
            return

        item = self.ui.hit_test(pos)
        if item is None:
            return

        self._select_cooldown = self._SELECT_COOLDOWN_FRAMES

        if item.action == "color":
            self.active_color = item.value
        elif item.action == "brush":
            self.brush_size = item.value
        elif item.action == "clear":
            self.engine.clear()
        elif item.action == "save":
            self._save_painting()

    def _handle_zoom_gesture(self, cursor_pos: Tuple[int, int]) -> None:
        """
        Handle the three-finger zoom gesture.

        When the user first raises three fingers (index+middle+ring), the current
        hand position is recorded as the "anchor". While holding the gesture:
        - Moving hand UP from anchor → zoom in
        - Moving hand DOWN from anchor → zoom out
        - Moving hand LEFT from anchor → pan left
        - Moving hand RIGHT from anchor → pan right

        A dead zone around the anchor prevents accidental zoom/pan from jitter.
        The anchor resets each time the gesture is released and re-engaged.
        """
        if not self._zoom_gesture_active:
            # First frame of zoom gesture: set the anchor point
            self._zoom_gesture_active = True
            self._zoom_anchor = cursor_pos
            self._zoom_direction = ""
            return

        if self._zoom_anchor is None:
            return

        # Calculate displacement from anchor
        dx = cursor_pos[0] - self._zoom_anchor[0]
        dy = cursor_pos[1] - self._zoom_anchor[1]

        # Determine dominant direction (only act outside dead zone)
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if max(abs_dx, abs_dy) < self._zoom_dead_zone:
            self._zoom_direction = ""
            return

        if self._zoom <= 1.01:
            # Not zoomed in yet: vertical movement controls zoom level
            if abs_dy > self._zoom_dead_zone:
                zoom_delta = -dy * self._zoom_sensitivity
                new_zoom = self._zoom + zoom_delta
                self._zoom = max(self._zoom_min, min(self._zoom_max, new_zoom))
                self._zoom_direction = "zoom_in" if dy < 0 else "zoom_out"
        else:
            # Already zoomed in: vertical = zoom, horizontal = pan
            # Apply both simultaneously for fluid control
            if abs_dy > self._zoom_dead_zone:
                zoom_delta = -dy * self._zoom_sensitivity
                new_zoom = self._zoom + zoom_delta
                self._zoom = max(self._zoom_min, min(self._zoom_max, new_zoom))

                if self._zoom <= 1.01:
                    self._pan_x = 0.0
                    self._pan_y = 0.0

            if abs_dx > self._zoom_dead_zone:
                pan_delta_x = dx * self._pan_sensitivity
                self._pan_x = max(
                    -(self.engine.width // 2),
                    min(self.engine.width // 2, self._pan_x + pan_delta_x),
                )

            if abs_dy > self._zoom_dead_zone and abs_dy <= abs_dx:
                # Vertical pan when horizontal movement dominates
                pan_delta_y = dy * self._pan_sensitivity
                self._pan_y = max(
                    -(self.engine.height // 2),
                    min(self.engine.height // 2, self._pan_y + pan_delta_y),
                )

            # Set direction label for cursor feedback
            if abs_dy > abs_dx:
                self._zoom_direction = "zoom_in" if dy < 0 else "zoom_out"
            else:
                self._zoom_direction = "pan_right" if dx > 0 else "pan_left"

        # Reset anchor to current position for continuous control
        self._zoom_anchor = cursor_pos

    def _finalize_drawing_state(self) -> None:
        """Clean up any in-progress drawing state."""
        if self._was_drawing:
            self.engine.end_stroke()
            self._was_drawing = False
            self._prev_draw_point = None

    def _draw_cursor(
        self,
        frame: np.ndarray,
        pos: Tuple[int, int],
        gesture: Gesture,
    ) -> None:
        """Draw a visual cursor for HoloPaint mode."""
        if gesture == Gesture.ZOOM_MODE or self._zoom_gesture_active:
            # Zoom mode: magnifying glass cursor with direction indicator
            color = (0, 220, 220)  # Cyan for zoom
            cx, cy = pos
            # Outer circle (lens)
            cv2.circle(frame, pos, 18, color, 2, cv2.LINE_AA)
            # Handle
            cv2.line(frame, (cx + 13, cy + 13), (cx + 22, cy + 22), color, 3, cv2.LINE_AA)
            # Plus/minus inside lens based on direction
            if self._zoom_direction == "zoom_in":
                cv2.line(frame, (cx - 8, cy), (cx + 8, cy), color, 2, cv2.LINE_AA)
                cv2.line(frame, (cx, cy - 8), (cx, cy + 8), color, 2, cv2.LINE_AA)
            elif self._zoom_direction == "zoom_out":
                cv2.line(frame, (cx - 8, cy), (cx + 8, cy), color, 2, cv2.LINE_AA)
            elif self._zoom_direction in ("pan_left", "pan_right"):
                # Arrow indicator for pan direction
                arrow_dir = -1 if self._zoom_direction == "pan_left" else 1
                cv2.arrowedLine(
                    frame, (cx - 6 * arrow_dir, cy), (cx + 6 * arrow_dir, cy),
                    color, 2, cv2.LINE_AA, tipLength=0.5,
                )
            # Label
            cv2.putText(
                frame, "ZOOM", (cx + 24, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
        elif self._fill_mode:
            # Fill mode: crosshair cursor
            size = 14
            color = (0, 200, 0)  # Green for fill mode
            cx, cy = pos
            cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2, cv2.LINE_AA)
            cv2.circle(frame, pos, size - 2, color, 1, cv2.LINE_AA)
            # "Fill" label next to cursor
            cv2.putText(
                frame, "FILL", (cx + size + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
        elif gesture == Gesture.DRAW:
            radius = max(self.brush_size.value // 2, 3)
            cv2.circle(frame, pos, radius, UI_CURSOR_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, pos, radius + 2, (0, 200, 100), 1, cv2.LINE_AA)
        elif gesture == Gesture.NAVIGATE:
            cv2.circle(frame, pos, 8, UI_CURSOR_NAV_COLOR, 2, cv2.LINE_AA)
            cv2.circle(frame, pos, 3, UI_CURSOR_NAV_COLOR, -1, cv2.LINE_AA)
        elif gesture == Gesture.SELECT:
            cv2.circle(frame, pos, 6, UI_CURSOR_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, pos, 10, UI_CURSOR_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pos, 8, UI_CURSOR_COLOR, 2, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Keyboard Input
    # -----------------------------------------------------------------------

    def _handle_keyboard(self) -> bool:
        """
        Handle keyboard input for all HoloPaint states.

        Returns:
            True if the application should quit, False otherwise.
        """
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            return True

        # --- Image selection state ---
        if self._state == self.STATE_IMAGE_SELECT:
            if key == ord('r') or key == ord('R'):
                self._fetch_random_image()
            elif key == ord('u') or key == ord('U'):
                self._upload_image()

        # --- Threshold adjustment state ---
        elif self._state == self.STATE_THRESHOLD_ADJUST:
            if key == ord('+') or key == ord('='):
                self.threshold = min(1.0, self.threshold + HOLOPAINT_THRESHOLD_STEP)
                self._process_current_image()
            elif key == ord('-') or key == ord('_'):
                self.threshold = max(0.05, self.threshold - HOLOPAINT_THRESHOLD_STEP)
                self._process_current_image()
            elif key == 13 or key == 10:  # Enter key
                self._start_painting()
            elif key == ord('n') or key == ord('N'):
                self._state = self.STATE_IMAGE_SELECT

        # --- Painting state ---
        elif self._state == self.STATE_PAINTING:
            if key == ord('n') or key == ord('N'):
                # Go back to image selection
                self._state = self.STATE_IMAGE_SELECT
                self.engine.clear()
                self._zoom = 1.0
                self._pan_x = 0.0
                self._pan_y = 0.0
            elif key == ord('s') or key == ord('S'):
                self._save_painting()
            elif key == ord('z') or key == ord('Z'):
                self.engine.undo()
            elif key == ord('y') or key == ord('Y'):
                self.engine.redo()
            elif key == ord('c') or key == ord('C'):
                self.engine.clear()
            elif key == ord('t') or key == ord('T'):
                # Return to threshold adjustment
                self._state = self.STATE_THRESHOLD_ADJUST
            elif key == ord('f') or key == ord('F'):
                # Toggle fill mode
                self._fill_mode = not self._fill_mode
                mode_name = "Fill" if self._fill_mode else "Paint"
                print(f"[HoloPaint] Switched to {mode_name} mode")

            # SPACE fills at last cursor position (reliable keyboard fill)
            elif key == 32:  # Space bar
                if self._fill_mode and self._last_cursor_pos is not None:
                    self._handle_fill(self._last_cursor_pos)

            # --- Zoom controls ---
            elif key == ord('i') or key == ord('I'):
                # Zoom in
                self._zoom = min(self._zoom_max, self._zoom + self._zoom_step)
            elif key == ord('o') or key == ord('O'):
                # Zoom out
                self._zoom = max(self._zoom_min, self._zoom - self._zoom_step)
                if self._zoom <= 1.01:
                    self._pan_x = 0.0
                    self._pan_y = 0.0

            # --- Pan controls (arrow keys) ---
            elif key == 81 or key == 2:  # Left arrow
                self._pan_x = max(self._pan_x - self._pan_step, -(self.engine.width // 2))
            elif key == 83 or key == 3:  # Right arrow
                self._pan_x = min(self._pan_x + self._pan_step, self.engine.width // 2)
            elif key == 82 or key == 0:  # Up arrow
                self._pan_y = max(self._pan_y - self._pan_step, -(self.engine.height // 2))
            elif key == 84 or key == 1:  # Down arrow
                self._pan_y = min(self._pan_y + self._pan_step, self.engine.height // 2)

            # Reset zoom
            elif key == ord('0'):
                self._zoom = 1.0
                self._pan_x = 0.0
                self._pan_y = 0.0

        return False

    # -----------------------------------------------------------------------
    # Image Acquisition
    # -----------------------------------------------------------------------

    def _fetch_random_image(self) -> None:
        """Fetch a random image from the internet and process it."""
        print("[HoloPaint] Fetching random image...")
        self.source_image = self.image_fetcher.fetch_random(640, 480)
        self._process_current_image()
        self._state = self.STATE_THRESHOLD_ADJUST

    def _upload_image(self) -> None:
        """
        Open a native file dialog to upload a local image.

        Uses macOS AppleScript (osascript) to open the native file picker,
        avoiding the tkinter/OpenCV NSApplication conflict that causes crashes
        on macOS when both GUI toolkits are active simultaneously.
        """
        import subprocess
        import sys

        filepath = None

        if sys.platform == "darwin":
            # macOS: use native file picker via AppleScript
            try:
                script = (
                    'set theFile to choose file with prompt '
                    '"Select an image for HoloPaint" '
                    'of type {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}\n'
                    'return POSIX path of theFile'
                )
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0 and result.stdout.strip():
                    filepath = result.stdout.strip()
                else:
                    print("[HoloPaint] No file selected.")
                    return
            except subprocess.TimeoutExpired:
                print("[HoloPaint] File dialog timed out.")
                return
            except Exception as e:
                print(f"[HoloPaint] File dialog error: {e}")
                return
        else:
            # Non-macOS: fall back to terminal input
            print("[HoloPaint] Enter image path in terminal:")
            filepath = input("Image path: ").strip()

        if filepath:
            image = self.image_fetcher.load_from_file(filepath)
            if image is not None:
                self.source_image = image
                self._process_current_image()
                self._state = self.STATE_THRESHOLD_ADJUST
            else:
                print("[HoloPaint] Failed to load the selected image.")

    def _start_painting(self) -> None:
        """Transition from threshold adjustment to painting mode."""
        if self.base_outline is not None:
            self._state = self.STATE_PAINTING
            self.engine.clear()  # Start with a fresh paint layer
            self._fit_outline_to_canvas()  # Pre-compute fitted outline
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._fill_mode = False
            self._zoom_gesture_active = False
            self._zoom_anchor = None
            self._zoom_direction = ""
            print("[HoloPaint] Painting mode active. Use gestures to paint!")
            print("[HoloPaint] Three fingers (index+middle+ring) = Zoom/Pan gesture")
            print("[HoloPaint] F=Toggle Fill | I/O=Zoom | Arrows=Pan | 0=Reset")
            print("[HoloPaint] N=New Image | S=Save | T=Adjust Threshold | Q=Quit")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def _save_painting(self) -> None:
        """
        Save the painted result (outlines + user paint) as a PNG file.
        """
        if self.base_outline is None or self.engine is None:
            return

        h, w = self.engine.height, self.engine.width

        # Start with the fitted outline (matching what's shown on screen)
        if self._fitted_outline is not None:
            result = self._fitted_outline.copy()
        else:
            result = np.full((h, w, 3), 255, dtype=np.uint8)

        # Overlay user's paint strokes
        canvas = self.engine.get_canvas()
        paint_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, paint_mask = cv2.threshold(paint_gray, 250, 255, cv2.THRESH_BINARY_INV)

        if np.any(paint_mask > 0):
            paint_mask_3ch = cv2.cvtColor(paint_mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (
                result * (1 - paint_mask_3ch) + canvas * paint_mask_3ch
            ).astype(np.uint8)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"holopaint_{timestamp}.png"
        success = cv2.imwrite(filepath, result)

        if success:
            print(f"[HoloPaint] Painting saved to: {filepath}")
        else:
            print(f"[HoloPaint] Error: Failed to save painting to {filepath}")

    # -----------------------------------------------------------------------
    # Mouse Input
    # -----------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for fallback drawing control."""
        if self._state != self.STATE_PAINTING:
            return

        cursor_pos = (x, y)
        self._last_cursor_pos = cursor_pos

        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            # Simulate SELECT or DRAW
            if cursor_pos[1] <= TOOLBAR_HEIGHT:
                self._handle_select(cursor_pos)
            elif self._fill_mode:
                self._handle_fill(cursor_pos)
            else:
                self._handle_draw(cursor_pos)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._mouse_down and not self._fill_mode and cursor_pos[1] > TOOLBAR_HEIGHT:
                self._handle_draw(cursor_pos)

        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False
            self._finalize_drawing_state()

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _update_fps(self) -> None:
        """Calculate and update the rolling FPS value."""
        current_time = time.time()
        elapsed = current_time - self._frame_time
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self._fps = self._fps * 0.9 + instant_fps * 0.1
        self._frame_time = current_time
