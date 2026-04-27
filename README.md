# HoloDraw - Hand Gesture Drawing Application

A real-time drawing application controlled through hand gestures, similar to MS Paint. Built with Python, MediaPipe (deep learning-based hand tracking), and OpenCV.

## Features

- **Freehand Drawing** - Draw freely with adjustable brush thickness
- **Geometric Shapes** - Line, Rectangle, Circle, and Ellipse tools
- **Eraser** - Remove portions of the drawing
- **Color Palette** - 10 preset colors available on the toolbar
- **Brush Sizes** - Small, Medium, and Large thickness options
- **Fill Toggle** - Switch between filled and outline shapes
- **Undo / Redo** - Full action history (up to 30 states)
- **Save to File** - Export drawings as timestamped PNG files
- **Clear Canvas** - Reset the entire drawing surface
- **Real-time FPS Display** - Performance monitoring in the status bar

## Requirements

- Python 3.8 or later
- A webcam (built-in or external)
- Adequate lighting for reliable hand detection

## Installation

1. Clone or download this repository.

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the MediaPipe HandLandmarker model (if not already bundled):
   ```bash
   curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```
   Place the `hand_landmarker.task` file in the project root directory alongside `main.py`.

## Usage

Run the application:
```bash
python main.py
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera INDEX` | 0 | Webcam device index |
| `--width WIDTH` | 1280 | Capture width in pixels |
| `--height HEIGHT` | 720 | Capture height in pixels |
| `--landmarks` | Off | Show hand landmark overlay for debugging |

### Example
```bash
python main.py --camera 0 --width 1920 --height 1080 --landmarks
```

## Gesture Guide

| Gesture | How To | Action |
|---------|--------|--------|
| **Index finger only** | Extend only the index finger | Draw on the canvas or place a shape |
| **Index + Middle fingers** | Extend index and middle fingers (peace sign) | Move cursor without drawing |
| **Pinch** | Touch thumb tip to index finger tip | Select a tool, color, or button on the toolbar |
| **Fist** | Close all fingers into a fist | Undo the last action |
| **Thumb up** | Extend only the thumb | Redo the last undone action |
| **Open palm** | Extend all five fingers | Idle / stop current action |

### Shape Placement Workflow

For shape tools (Line, Rectangle, Circle, Ellipse):
1. Select the shape tool using a pinch gesture on the toolbar.
2. Point with the index finger to set the **start point**.
3. Move the index finger to the desired **end point** (a live preview will be shown).
4. Switch to the **navigate gesture** (index + middle fingers) to finalize the shape.

## Keyboard Shortcuts

These keyboard shortcuts serve as backup controls:

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save the current drawing |
| `Z` | Undo |
| `Y` | Redo |
| `C` | Clear the canvas |
| `F` | Toggle fill mode |
| `L` | Toggle landmark display |

## Project Structure

```
HoloDraw/
├── main.py                 # Application entry point
├── config.py               # Constants, enums, and settings
├── hand_tracker.py         # MediaPipe hand landmark detection
├── gesture_recognizer.py   # Gesture classification from landmarks
├── drawing_engine.py       # Canvas management and undo/redo
├── shape_tools.py          # Shape drawing implementations
├── ui_overlay.py           # Toolbar and status bar rendering
├── app_controller.py       # Main application loop and state machine
├── hand_landmarker.task    # MediaPipe HandLandmarker DL model (float16)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Architecture

The application follows a modular pipeline architecture:

1. **HandTracker** captures frames and detects 21 hand landmarks using the MediaPipe Hands deep learning model.
2. **GestureRecognizer** classifies the hand pose into a gesture using geometric rules on the landmark positions.
3. **AppController** maps gestures to drawing actions and manages application state.
4. **DrawingEngine** maintains the canvas, executes draw operations, and manages the undo/redo history.
5. **UIOverlay** renders the toolbar, color palette, and status bar onto the display frame.

## Troubleshooting

- **Hand not detected**: Ensure adequate, even lighting. Avoid strong backlighting or very dark environments.
- **Erratic gesture switching**: Keep your hand clearly visible and make deliberate gestures. The system includes debouncing to reduce accidental switches.
- **Low FPS**: Try reducing the capture resolution with `--width 640 --height 480`.
- **Camera not opening**: Check that no other application is using the webcam. Try a different `--camera` index.

## License

This project is provided for educational and personal use.
