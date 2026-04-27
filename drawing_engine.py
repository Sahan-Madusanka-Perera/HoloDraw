"""
HoloDraw Drawing Engine Module
--------------------------------
Manages the drawing canvas, undo/redo history, and all draw operations.
The canvas is maintained as a NumPy array with a white background. Every
meaningful drawing action (freehand stroke segment, shape commit, erase)
is recorded so that undo and redo operations work reliably.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from config import (
    Tool, BrushSize,
    CANVAS_BG_COLOR, MAX_UNDO_HISTORY, ERASER_RADIUS,
    COLOR_WHITE,
)
from shape_tools import SHAPE_DRAW_FUNCTIONS


class DrawingEngine:
    """
    Core drawing engine that owns the canvas state and history stack.

    The canvas is a BGR NumPy array initialized to the configured background
    color. Drawing operations modify this array directly. Before each new
    logical action (e.g., starting a freehand stroke, committing a shape),
    the current canvas state is pushed onto the undo stack.

    Attributes:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        canvas: The current drawing surface (NumPy array).
        _undo_stack: List of previous canvas states for undo.
        _redo_stack: List of undone canvas states for redo.
    """

    def __init__(self, width: int, height: int):
        """
        Create a new drawing engine with a blank canvas.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
        """
        self.width = width
        self.height = height
        self.canvas = self._create_blank_canvas()
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._is_stroke_active: bool = False

    def _create_blank_canvas(self) -> np.ndarray:
        """Create a new blank canvas filled with the background color."""
        canvas = np.full(
            (self.height, self.width, 3),
            fill_value=CANVAS_BG_COLOR,
            dtype=np.uint8,
        )
        return canvas

    def begin_stroke(self) -> None:
        """
        Mark the beginning of a new freehand stroke.

        Saves the current canvas state to the undo stack so that the entire
        stroke can be undone as a single action. This should be called once
        when the user transitions from NAVIGATE to DRAW gesture.
        """
        if not self._is_stroke_active:
            self._push_undo()
            self._is_stroke_active = True

    def end_stroke(self) -> None:
        """Mark the end of the current freehand stroke."""
        self._is_stroke_active = False

    def draw_freehand(
        self,
        prev_point: Tuple[int, int],
        curr_point: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Draw a single segment of a freehand stroke.

        Connects the previous cursor position to the current one with a
        smooth anti-aliased line. The caller is responsible for calling
        begin_stroke() before the first segment and end_stroke() after
        the last segment.

        Args:
            prev_point: (x, y) of the previous cursor position.
            curr_point: (x, y) of the current cursor position.
            color: BGR color tuple.
            thickness: Stroke width in pixels.
        """
        cv2.line(
            self.canvas, prev_point, curr_point, color, thickness,
            lineType=cv2.LINE_AA,
        )

    def commit_shape(
        self,
        tool: Tool,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        filled: bool = False,
    ) -> None:
        """
        Finalize a shape onto the canvas.

        Pushes the current canvas state to the undo stack before drawing,
        so the shape placement can be undone as a single action.

        Args:
            tool: The shape tool type (LINE, RECTANGLE, CIRCLE, ELLIPSE).
            start: (x, y) anchor point of the shape.
            end: (x, y) end point of the shape.
            color: BGR color tuple.
            thickness: Border width in pixels.
            filled: Whether to fill the shape (only for rect, circle, ellipse).
        """
        draw_fn = SHAPE_DRAW_FUNCTIONS.get(tool)
        if draw_fn is None:
            return

        self._push_undo()

        # Line drawing function does not accept 'filled' parameter
        if tool == Tool.LINE:
            draw_fn(self.canvas, start, end, color, thickness)
        else:
            draw_fn(self.canvas, start, end, color, thickness, filled)

    def erase(self, point: Tuple[int, int], radius: int = ERASER_RADIUS) -> None:
        """
        Erase a circular region around the given point by painting it
        with the canvas background color.

        Args:
            point: (x, y) center of the eraser.
            radius: Radius of the eraser circle in pixels.
        """
        cv2.circle(self.canvas, point, radius, CANVAS_BG_COLOR, -1)

    def begin_erase(self) -> None:
        """
        Save the canvas state before an erase sequence begins.
        Similar to begin_stroke() but for eraser operations.
        """
        if not self._is_stroke_active:
            self._push_undo()
            self._is_stroke_active = True

    def undo(self) -> bool:
        """
        Revert the canvas to the previous state.

        Returns:
            True if an undo was performed, False if the stack is empty.
        """
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.canvas.copy())
        self.canvas = self._undo_stack.pop()
        self._is_stroke_active = False
        return True

    def redo(self) -> bool:
        """
        Re-apply the last undone action.

        Returns:
            True if a redo was performed, False if the stack is empty.
        """
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.canvas.copy())
        self.canvas = self._redo_stack.pop()
        return True

    def clear(self) -> None:
        """Clear the entire canvas to the background color."""
        self._push_undo()
        self.canvas = self._create_blank_canvas()
        self._redo_stack.clear()

    def save(self, filepath: str) -> bool:
        """
        Export the current canvas as an image file.

        Args:
            filepath: Output file path (e.g., "drawing.png").

        Returns:
            True if the save succeeded, False otherwise.
        """
        return cv2.imwrite(filepath, self.canvas)

    def get_canvas(self) -> np.ndarray:
        """Return the current canvas array (read-only reference)."""
        return self.canvas

    def _push_undo(self) -> None:
        """
        Push the current canvas state onto the undo stack.

        Enforces the maximum history size by discarding the oldest entry
        when the limit is reached. Clears the redo stack because a new
        action invalidates the redo timeline.
        """
        if len(self._undo_stack) >= MAX_UNDO_HISTORY:
            self._undo_stack.pop(0)
        self._undo_stack.append(self.canvas.copy())
        self._redo_stack.clear()
