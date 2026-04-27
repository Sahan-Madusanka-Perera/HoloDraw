"""
HoloDraw Shape Tools Module
-----------------------------
Provides static functions for drawing geometric shapes on the canvas.
Each function operates directly on a NumPy array (the canvas) using OpenCV
drawing primitives. Also includes a preview function for rendering
semi-transparent shape outlines during the placement workflow.
"""

import cv2
import numpy as np
from typing import Tuple

from config import Tool


def draw_line(
    canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """
    Draw a straight line between two points.

    Args:
        canvas: The drawing surface (modified in place).
        start: (x, y) starting pixel coordinate.
        end: (x, y) ending pixel coordinate.
        color: BGR color tuple.
        thickness: Line width in pixels.
    """
    cv2.line(canvas, start, end, color, thickness, lineType=cv2.LINE_AA)


def draw_rectangle(
    canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    filled: bool = False,
) -> None:
    """
    Draw a rectangle defined by two corner points.

    Args:
        canvas: The drawing surface (modified in place).
        start: (x, y) of one corner.
        end: (x, y) of the opposite corner.
        color: BGR color tuple.
        thickness: Border width in pixels (ignored when filled).
        filled: If True, the rectangle is filled with the given color.
    """
    fill = -1 if filled else thickness
    cv2.rectangle(canvas, start, end, color, fill, lineType=cv2.LINE_AA)


def draw_circle(
    canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    filled: bool = False,
) -> None:
    """
    Draw a circle using the start point as the center and the distance
    to the end point as the radius.

    Args:
        canvas: The drawing surface (modified in place).
        start: (x, y) center of the circle.
        end: (x, y) point on the circumference (defines the radius).
        color: BGR color tuple.
        thickness: Border width in pixels (ignored when filled).
        filled: If True, the circle is filled with the given color.
    """
    radius = int(np.hypot(end[0] - start[0], end[1] - start[1]))
    radius = max(radius, 1)
    fill = -1 if filled else thickness
    cv2.circle(canvas, start, radius, color, fill, lineType=cv2.LINE_AA)


def draw_ellipse(
    canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    filled: bool = False,
) -> None:
    """
    Draw an axis-aligned ellipse inscribed in the bounding box defined
    by the start and end points.

    Args:
        canvas: The drawing surface (modified in place).
        start: (x, y) of one corner of the bounding box.
        end: (x, y) of the opposite corner of the bounding box.
        color: BGR color tuple.
        thickness: Border width in pixels (ignored when filled).
        filled: If True, the ellipse is filled with the given color.
    """
    center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    axes = (abs(end[0] - start[0]) // 2, abs(end[1] - start[1]) // 2)
    axes = (max(axes[0], 1), max(axes[1], 1))
    fill = -1 if filled else thickness
    cv2.ellipse(canvas, center, axes, 0, 0, 360, color, fill, lineType=cv2.LINE_AA)


def preview_shape(
    display: np.ndarray,
    shape_type: Tool,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """
    Render a non-destructive preview of a shape on a copy of the display frame.

    This is called every frame during shape placement so the user can see
    where the shape will be drawn before committing. The preview is drawn
    with a thinner line and does not modify the actual canvas.

    Args:
        display: The current display frame (not modified; a copy is returned).
        shape_type: The type of shape being placed.
        start: (x, y) starting anchor point.
        end: (x, y) current cursor position.
        color: BGR color tuple for the preview outline.
        thickness: Preview line thickness.

    Returns:
        A copy of display with the shape preview drawn on it.
    """
    preview = display.copy()

    if shape_type == Tool.LINE:
        cv2.line(preview, start, end, color, thickness, lineType=cv2.LINE_AA)

    elif shape_type == Tool.RECTANGLE:
        cv2.rectangle(preview, start, end, color, thickness, lineType=cv2.LINE_AA)

    elif shape_type == Tool.CIRCLE:
        radius = int(np.hypot(end[0] - start[0], end[1] - start[1]))
        radius = max(radius, 1)
        cv2.circle(preview, start, radius, color, thickness, lineType=cv2.LINE_AA)

    elif shape_type == Tool.ELLIPSE:
        center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        axes = (abs(end[0] - start[0]) // 2, abs(end[1] - start[1]) // 2)
        axes = (max(axes[0], 1), max(axes[1], 1))
        cv2.ellipse(
            preview, center, axes, 0, 0, 360, color, thickness, lineType=cv2.LINE_AA
        )

    return preview


# Dispatch table for finalizing shapes on the canvas
SHAPE_DRAW_FUNCTIONS = {
    Tool.LINE: draw_line,
    Tool.RECTANGLE: draw_rectangle,
    Tool.CIRCLE: draw_circle,
    Tool.ELLIPSE: draw_ellipse,
}
