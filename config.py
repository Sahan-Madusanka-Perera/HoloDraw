"""
HoloDraw Configuration Module
-----------------------------
Centralizes all application constants, enumerations, and default settings.
All tunable parameters are defined here to avoid magic numbers throughout the codebase.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Tool(Enum):
    """Available drawing tools."""
    FREEHAND = auto()
    LINE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()
    ELLIPSE = auto()
    ERASER = auto()


class Gesture(Enum):
    """Recognized hand gestures mapped to application actions."""
    IDLE = auto()       # No actionable gesture detected
    DRAW = auto()       # Index finger extended only -> draw / place shape
    NAVIGATE = auto()   # Index + middle fingers extended -> move without drawing
    SELECT = auto()     # Pinch (thumb + index close together) -> toolbar selection
    UNDO = auto()       # Fist (all fingers curled) -> undo last action
    REDO = auto()       # Thumb up only -> redo last undone action


class BrushSize(Enum):
    """Predefined brush thickness levels."""
    SMALL = 3
    MEDIUM = 7
    LARGE = 14


# ---------------------------------------------------------------------------
# Color Definitions (BGR format for OpenCV)
# ---------------------------------------------------------------------------

# Drawing palette colors
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (60, 60, 220)
COLOR_GREEN = (80, 180, 80)
COLOR_BLUE = (200, 120, 50)
COLOR_YELLOW = (50, 220, 240)
COLOR_ORANGE = (40, 140, 255)
COLOR_PURPLE = (180, 60, 160)
COLOR_CYAN = (210, 200, 50)
COLOR_PINK = (170, 120, 230)

# Ordered palette for the toolbar
COLOR_PALETTE = [
    COLOR_BLACK,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_CYAN,
    COLOR_PINK,
    COLOR_WHITE,
]

# UI theme colors
UI_BG_COLOR = (32, 28, 24)             # Deep slate toolbar background (#181C20)
UI_BG_COLOR_ALT = (45, 40, 35)         # Lighter sleek background (#23282D)
UI_BORDER_COLOR = (70, 60, 55)         # Subtle precise borders
UI_HIGHLIGHT_COLOR = (255, 230, 0)     # Electric Cyan active highlight (#00E6FF)
UI_TEXT_COLOR = (255, 255, 255)        # Pure white primary text
UI_TEXT_DIM_COLOR = (170, 170, 170)    # Crisp secondary text
UI_CURSOR_COLOR = (200, 50, 255)       # Neon pink on-screen cursor
UI_CURSOR_NAV_COLOR = (255, 150, 0)    # Neon blue navigation cursor
UI_STATUS_BG = (22, 18, 15)            # Very dark status bar background


# ---------------------------------------------------------------------------
# MediaPipe Hand Landmark Indices (reference for readability)
# ---------------------------------------------------------------------------

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


# ---------------------------------------------------------------------------
# Gesture Recognition Thresholds
# ---------------------------------------------------------------------------

# Minimum Euclidean distance (in pixels) below which thumb-tip and index-tip
# are considered to be in a "pinch" gesture.
PINCH_DISTANCE_THRESHOLD = 55

# Number of consecutive frames a gesture must persist before it is accepted.
# Prevents erratic gesture switching caused by brief misdetections.
GESTURE_DEBOUNCE_FRAMES = 2

# Cooldown frames after an undo/redo gesture is triggered to prevent
# repeated rapid-fire undo/redo from a single held gesture.
UNDO_REDO_COOLDOWN_FRAMES = 15


# ---------------------------------------------------------------------------
# Canvas and Display Settings
# ---------------------------------------------------------------------------

# Default webcam capture resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# The toolbar occupies the top portion of the display
TOOLBAR_HEIGHT = 80

# Status bar at the bottom of the display
STATUS_BAR_HEIGHT = 36

# Maximum number of canvas states kept in the undo history
MAX_UNDO_HISTORY = 30

# Canvas background color (white, like a blank page)
CANVAS_BG_COLOR = COLOR_WHITE

# Opacity of the canvas overlay on top of the camera feed (0.0 - 1.0).
# 0.0 = fully transparent (camera only), 1.0 = fully opaque (canvas only).
CANVAS_OPACITY = 0.45

# Eraser radius (pixels)
ERASER_RADIUS = 28


# ---------------------------------------------------------------------------
# UI Layout Constants
# ---------------------------------------------------------------------------

# Tool button dimensions in the toolbar
TOOL_BUTTON_SIZE = 50
TOOL_BUTTON_MARGIN = 10
TOOL_BUTTON_Y_OFFSET = 15

# Color swatch dimensions in the toolbar
COLOR_SWATCH_RADIUS = 14
COLOR_SWATCH_MARGIN = 8
COLOR_SWATCH_Y_CENTER = 40

# Brush size indicator position
BRUSH_SIZE_Y_CENTER = 40

# Minimum distance (pixels) the cursor must move between frames to register
# as intentional movement and trigger freehand drawing. Reduces jitter.
MIN_DRAW_DISTANCE = 3


# ---------------------------------------------------------------------------
# HoloPaint Settings
# ---------------------------------------------------------------------------

# Path to the trained U-Net edge detection model weights
EDGE_MODEL_PATH = "holopaint_unet.pth"

# Default edge detection threshold (0.0–1.0).
# Lower values produce more edges; higher values produce fewer, stronger edges.
HOLOPAINT_DEFAULT_THRESHOLD = 0.3

# Step size for threshold adjustment via keyboard (+/- keys)
HOLOPAINT_THRESHOLD_STEP = 0.05

# Opacity of the edge outline overlay on the camera feed (0.0–1.0)
HOLOPAINT_OUTLINE_OPACITY = 0.9

# Opacity of user's paint strokes over the outline (0.0–1.0)
HOLOPAINT_PAINT_OPACITY = 0.85


# ---------------------------------------------------------------------------
# Application Settings
# ---------------------------------------------------------------------------

@dataclass
class AppSettings:
    """Runtime application settings with sensible defaults."""
    camera_index: int = 0
    camera_width: int = CAMERA_WIDTH
    camera_height: int = CAMERA_HEIGHT
    window_name: str = "HoloDraw - Hand Gesture Drawing"
    show_landmarks: bool = False        # Toggle hand landmark overlay for debugging
    mirror_mode: bool = True            # Flip frame horizontally for natural interaction
    detection_confidence: float = 0.7   # MediaPipe detection confidence threshold
    tracking_confidence: float = 0.7    # MediaPipe tracking confidence threshold
