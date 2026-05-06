"""
HoloDraw UI Overlay Module
----------------------------
Renders the toolbar, color palette, brush size indicators, and status bar
directly onto the display frame using OpenCV drawing primitives.
No external image assets are required; all icons are drawn programmatically.

Layout (top toolbar, left to right):
    [Tool Buttons] | [Color Palette] | [Brush Sizes] | [Fill Toggle]
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

from config import (
    Tool, BrushSize,
    TOOLBAR_HEIGHT, STATUS_BAR_HEIGHT,
    TOOL_BUTTON_SIZE, TOOL_BUTTON_MARGIN, TOOL_BUTTON_Y_OFFSET,
    COLOR_PALETTE, COLOR_SWATCH_RADIUS, COLOR_SWATCH_MARGIN, COLOR_SWATCH_Y_CENTER,
    BRUSH_SIZE_Y_CENTER,
    UI_BG_COLOR, UI_BG_COLOR_ALT, UI_BORDER_COLOR,
    UI_HIGHLIGHT_COLOR, UI_TEXT_COLOR, UI_TEXT_DIM_COLOR, UI_STATUS_BG,
    COLOR_WHITE, COLOR_BLACK, ERASER_RADIUS,
)


class ToolbarItem:
    """Represents a clickable region in the toolbar."""

    def __init__(self, x: int, y: int, w: int, h: int, action: str, value=None):
        """
        Args:
            x, y: Top-left corner of the clickable region.
            w, h: Width and height of the clickable region.
            action: Type of action ("tool", "color", "brush", "fill", "clear", "save").
            value: Associated value (Tool enum, color tuple, BrushSize, etc.).
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.action = action
        self.value = value

    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if a point falls within this toolbar item's bounds."""
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


class UIOverlay:
    """
    Manages rendering and hit-testing for all on-screen UI elements.

    The toolbar is constructed once during initialization (positions are
    computed and stored) and re-rendered each frame. Hit-testing allows
    the application controller to determine which toolbar element, if any,
    was targeted by a pinch gesture.
    """

    def __init__(self, frame_width: int, frame_height: int):
        """
        Compute toolbar element positions based on the frame dimensions.

        Args:
            frame_width: Width of the display frame in pixels.
            frame_height: Height of the display frame in pixels.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._items: List[ToolbarItem] = []
        self._build_layout()

    def _build_layout(self) -> None:
        """Compute positions for all toolbar elements and store them."""
        self._items.clear()
        x_cursor = TOOL_BUTTON_MARGIN

        # --- Tool buttons ---
        tools = [
            Tool.FREEHAND, Tool.LINE, Tool.RECTANGLE,
            Tool.CIRCLE, Tool.ELLIPSE, Tool.ERASER,
        ]
        for tool in tools:
            self._items.append(ToolbarItem(
                x=x_cursor,
                y=TOOL_BUTTON_Y_OFFSET,
                w=TOOL_BUTTON_SIZE,
                h=TOOL_BUTTON_SIZE,
                action="tool",
                value=tool,
            ))
            x_cursor += TOOL_BUTTON_SIZE + TOOL_BUTTON_MARGIN

        # Separator space
        x_cursor += TOOL_BUTTON_MARGIN * 2

        # --- Color palette ---
        swatch_diameter = COLOR_SWATCH_RADIUS * 2
        for color in COLOR_PALETTE:
            self._items.append(ToolbarItem(
                x=x_cursor,
                y=COLOR_SWATCH_Y_CENTER - COLOR_SWATCH_RADIUS,
                w=swatch_diameter,
                h=swatch_diameter,
                action="color",
                value=color,
            ))
            x_cursor += swatch_diameter + COLOR_SWATCH_MARGIN

        # Separator space
        x_cursor += TOOL_BUTTON_MARGIN * 2

        # --- Brush size buttons ---
        for size in BrushSize:
            self._items.append(ToolbarItem(
                x=x_cursor,
                y=BRUSH_SIZE_Y_CENTER - 15,
                w=34,
                h=30,
                action="brush",
                value=size,
            ))
            x_cursor += 34 + TOOL_BUTTON_MARGIN

        # Separator space
        x_cursor += TOOL_BUTTON_MARGIN

        # --- Fill toggle button ---
        self._items.append(ToolbarItem(
            x=x_cursor,
            y=TOOL_BUTTON_Y_OFFSET,
            w=TOOL_BUTTON_SIZE,
            h=TOOL_BUTTON_SIZE,
            action="fill",
            value=None,
        ))
        x_cursor += TOOL_BUTTON_SIZE + TOOL_BUTTON_MARGIN * 2

        # --- Clear button ---
        self._items.append(ToolbarItem(
            x=x_cursor,
            y=TOOL_BUTTON_Y_OFFSET,
            w=TOOL_BUTTON_SIZE,
            h=TOOL_BUTTON_SIZE,
            action="clear",
            value=None,
        ))
        x_cursor += TOOL_BUTTON_SIZE + TOOL_BUTTON_MARGIN

        # --- Save button ---
        self._items.append(ToolbarItem(
            x=x_cursor,
            y=TOOL_BUTTON_Y_OFFSET,
            w=TOOL_BUTTON_SIZE,
            h=TOOL_BUTTON_SIZE,
            action="save",
            value=None,
        ))

    def render(
        self,
        frame: np.ndarray,
        active_tool: Tool,
        active_color: Tuple[int, int, int],
        brush_size: BrushSize,
        filled: bool,
        fps: float,
        hand_detected: bool,
        gesture_name: str = "",
    ) -> np.ndarray:
        """
        Draw the complete UI overlay onto the display frame.

        Args:
            frame: The display frame to draw on (modified in place).
            active_tool: Currently selected drawing tool.
            active_color: Currently selected drawing color (BGR).
            brush_size: Current brush size setting.
            filled: Whether shapes are drawn filled.
            fps: Current frames per second for the status bar.
            hand_detected: Whether a hand is currently detected.
            gesture_name: Name of the current gesture for the status bar.

        Returns:
            The frame with UI elements rendered on it.
        """
        # Draw glassmorphism floating toolbar background
        tb_margin = 10
        overlay = frame.copy()
        
        # Base glass pill
        self._draw_rounded_rect(
            overlay, 
            (tb_margin, tb_margin), 
            (self.frame_width - tb_margin, TOOLBAR_HEIGHT - tb_margin), 
            UI_BG_COLOR, -1, 16
        )
        
        # Blend for transparency
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Sleek glossy border
        self._draw_rounded_rect(
            frame, 
            (tb_margin, tb_margin), 
            (self.frame_width - tb_margin, TOOLBAR_HEIGHT - tb_margin), 
            UI_BORDER_COLOR, 1, 16
        )


        # Render each toolbar item
        for item in self._items:
            if item.action == "tool":
                self._render_tool_button(frame, item, active_tool)
            elif item.action == "color":
                self._render_color_swatch(frame, item, active_color)
            elif item.action == "brush":
                self._render_brush_button(frame, item, brush_size)
            elif item.action == "fill":
                self._render_fill_toggle(frame, item, filled)
            elif item.action == "clear":
                self._render_action_button(frame, item, "CLR", (100, 100, 220))
            elif item.action == "save":
                self._render_action_button(frame, item, "SAV", (100, 180, 100))

        # Draw status bar
        self._render_status_bar(frame, fps, hand_detected, gesture_name,
                                 active_tool, active_color)

        return frame

    def hit_test(self, point: Tuple[int, int]) -> Optional[ToolbarItem]:
        """
        Determine which toolbar item (if any) contains the given point.

        Args:
            point: (x, y) pixel coordinate to test.

        Returns:
            The matching ToolbarItem, or None if the point is outside all items.
        """
        for item in self._items:
            if item.contains(point):
                return item
        return None

    # --- Private rendering methods ---

    def _draw_rounded_rect(
        self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
        color: Tuple[int, int, int], thickness: int, r: int,
    ) -> None:
        """Helper to draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        if thickness < 0:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
        else:
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def _render_tool_button(
        self,
        frame: np.ndarray,
        item: ToolbarItem,
        active_tool: Tool,
    ) -> None:
        """Draw a tool selection button with an icon representing the tool."""
        x, y, w, h = item.x, item.y, item.w, item.h
        is_active = (item.value == active_tool)

        # Button background
        bg = UI_BG_COLOR_ALT if is_active else UI_BG_COLOR
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), bg, -1, 10)

        # Active highlight border
        if is_active:
            self._draw_rounded_rect(frame, (x, y), (x + w, y + h), UI_HIGHLIGHT_COLOR, 2, 10)
        else:
            self._draw_rounded_rect(frame, (x, y), (x + w, y + h), UI_BORDER_COLOR, 1, 10)

        # Draw tool icon (simple geometric representations)
        cx, cy = x + w // 2, y + h // 2
        icon_color = UI_HIGHLIGHT_COLOR if is_active else UI_TEXT_COLOR

        if item.value == Tool.FREEHAND:
            # Wavy line to represent freehand drawing
            pts = np.array([
                [x + 10, cy + 8], [x + 18, cy - 8],
                [x + 26, cy + 4], [x + 34, cy - 6], [x + 40, cy + 2]
            ], np.int32)
            cv2.polylines(frame, [pts], False, icon_color, 2, cv2.LINE_AA)

        elif item.value == Tool.LINE:
            cv2.line(frame, (x + 10, y + h - 10), (x + w - 10, y + 10),
                     icon_color, 2, cv2.LINE_AA)

        elif item.value == Tool.RECTANGLE:
            cv2.rectangle(frame, (x + 10, y + 12), (x + w - 10, y + h - 12),
                          icon_color, 2)

        elif item.value == Tool.CIRCLE:
            cv2.circle(frame, (cx, cy), min(w, h) // 2 - 10, icon_color, 2,
                       cv2.LINE_AA)

        elif item.value == Tool.ELLIPSE:
            cv2.ellipse(frame, (cx, cy), (w // 2 - 10, h // 2 - 14),
                        0, 0, 360, icon_color, 2, cv2.LINE_AA)

        elif item.value == Tool.ERASER:
            # Eraser icon: a filled rounded rectangle
            cv2.rectangle(frame, (x + 12, y + 14), (x + w - 12, y + h - 10),
                          icon_color, 2)
            cv2.line(frame, (x + 12, y + 22), (x + w - 12, y + 22),
                     icon_color, 1)

    def _render_color_swatch(
        self,
        frame: np.ndarray,
        item: ToolbarItem,
        active_color: Tuple[int, int, int],
    ) -> None:
        """Draw a circular color swatch with a highlight if it is active."""
        cx = item.x + COLOR_SWATCH_RADIUS
        cy = item.y + COLOR_SWATCH_RADIUS
        is_active = (item.value == active_color)

        # Draw filled circle for the color
        cv2.circle(frame, (cx, cy), COLOR_SWATCH_RADIUS, item.value, -1,
                   cv2.LINE_AA)

        # Highlight ring for the active color
        if is_active:
            cv2.circle(frame, (cx, cy), COLOR_SWATCH_RADIUS + 3,
                       UI_HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), COLOR_SWATCH_RADIUS,
                       UI_BORDER_COLOR, 1, cv2.LINE_AA)

    def _render_brush_button(
        self,
        frame: np.ndarray,
        item: ToolbarItem,
        active_size: BrushSize,
    ) -> None:
        """Draw a brush size button displaying a dot of the corresponding size."""
        x, y, w, h = item.x, item.y, item.w, item.h
        is_active = (item.value == active_size)

        # Background
        bg = UI_BG_COLOR_ALT if is_active else UI_BG_COLOR
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), bg, -1, 8)

        if is_active:
            self._draw_rounded_rect(frame, (x, y), (x + w, y + h), UI_HIGHLIGHT_COLOR, 2, 8)
        else:
            self._draw_rounded_rect(frame, (x, y), (x + w, y + h), UI_BORDER_COLOR, 1, 8)

        # Dot size proportional to the brush value
        dot_radius = max(item.value.value // 2, 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), dot_radius, UI_TEXT_COLOR, -1, cv2.LINE_AA)

    def _render_fill_toggle(
        self,
        frame: np.ndarray,
        item: ToolbarItem,
        filled: bool,
    ) -> None:
        """Draw the fill/outline toggle button."""
        x, y, w, h = item.x, item.y, item.w, item.h

        # Background
        bg = UI_BG_COLOR_ALT if filled else UI_BG_COLOR
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), bg, -1, 10)

        border_color = UI_HIGHLIGHT_COLOR if filled else UI_BORDER_COLOR
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), border_color, 2 if filled else 1, 10)

        # Icon: filled vs outline square
        icon_color = UI_HIGHLIGHT_COLOR if filled else UI_TEXT_COLOR
        fill_val = -1 if filled else 2
        cv2.rectangle(frame, (x + 12, y + 14), (x + w - 12, y + h - 14),
                      icon_color, fill_val)

    def _render_action_button(
        self,
        frame: np.ndarray,
        item: ToolbarItem,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw an action button (clear, save) with a text label."""
        x, y, w, h = item.x, item.y, item.w, item.h
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), UI_BG_COLOR, -1, 10)
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), UI_BORDER_COLOR, 1, 10)

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        tx = x + (w - text_size[0]) // 2
        ty = y + (h + text_size[1]) // 2
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)

    def _render_status_bar(
        self,
        frame: np.ndarray,
        fps: float,
        hand_detected: bool,
        gesture_name: str,
        active_tool: Tool,
        active_color: Tuple[int, int, int],
    ) -> None:
        """Draw the status bar at the bottom of the frame."""
        bar_y = self.frame_height - STATUS_BAR_HEIGHT
        cv2.rectangle(frame, (0, bar_y), (self.frame_width, self.frame_height),
                       UI_STATUS_BG, -1)
        cv2.line(frame, (0, bar_y), (self.frame_width, bar_y), UI_BORDER_COLOR, 1)

        text_y = bar_y + STATUS_BAR_HEIGHT - 10

        # FPS
        fps_text = f"FPS: {fps:.0f}"
        cv2.putText(frame, fps_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, UI_TEXT_DIM_COLOR, 1, cv2.LINE_AA)

        # Hand status
        hand_status = "Hand: Detected" if hand_detected else "Hand: Not Detected"
        hand_color = (0, 200, 100) if hand_detected else (0, 80, 200)
        cv2.putText(frame, hand_status, (120, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, hand_color, 1, cv2.LINE_AA)

        # Current gesture
        if gesture_name:
            cv2.putText(frame, f"Gesture: {gesture_name}", (310, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_TEXT_COLOR, 1, cv2.LINE_AA)

        # Active tool
        tool_text = f"Tool: {active_tool.name.capitalize()}"
        cv2.putText(frame, tool_text, (520, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, UI_TEXT_COLOR, 1, cv2.LINE_AA)

        # Active color indicator
        cv2.circle(frame, (700, text_y - 5), 8, active_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (700, text_y - 5), 8, UI_BORDER_COLOR, 1, cv2.LINE_AA)

        # Keyboard shortcuts hint
        hint = "Q:Quit | S:Save | Z:Undo | Y:Redo | C:Clear"
        cv2.putText(frame, hint, (740, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, UI_TEXT_DIM_COLOR, 1, cv2.LINE_AA)
