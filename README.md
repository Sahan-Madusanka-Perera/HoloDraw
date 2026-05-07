# HoloFun - Hand Gesture Drawing & Painting Application

A real-time, interactive drawing and colouring application controlled through hand gestures. Built with Python, MediaPipe (AI-based hand tracking), and a custom-trained U-Net edge detection model.

## Modes

### 1. HoloDraw — Freehand Gesture Drawing
Draw freely on a digital canvas using hand gestures. Supports freehand sketching, geometric shapes, colors, brush sizes, and full undo/redo history.

### 2. HoloPaint — AI-Powered Paint-Over Outlines
Fetch a random image from the internet or upload your own, then a **trained U-Net deep learning model** extracts the edges/outlines. Paint over these outlines using hand gestures — like a digital coloring book!

## Features

### HoloDraw Features
- **Freehand Drawing** - Draw freely with adjustable brush thickness
- **Geometric Shapes** - Line, Rectangle, Circle, and Ellipse tools
- **Eraser** - Remove portions of the drawing
- **Color Palette** - 20 distinct preset colors available on a 2-row toolbar
- **Brush Sizes** - Small, Medium, and Large thickness options
- **Fill Toggle** - Switch between filled and outline shapes
- **Undo / Redo** - Full action history (up to 30 states)
- **Zoom & Pan** - Gesture-controlled 5x zoom and panning for high-detail work
- **Real-time FPS Display** - Performance monitoring in the status bar

### HoloPaint Features
- **Random Image Fetch** - Get random photos from Lorem Picsum (no API key needed)
- **Image Upload** - Load any image from your computer
- **Advanced Edge Extraction** - Multi-scale tiled inference + Hybrid Canny fusion
- **Gap Bridging** - Skeleton-aware endpoint bridging to ensure closed, paintable regions
- **Two-Layer Canvas** - Base outlines (non-erasable) + user paint overlay
- **Flood Fill** - One-tap region filling using the SELECT gesture

## Requirements

- Python 3.8 or later
- A webcam (built-in or external)
- Adequate lighting for reliable hand detection
- PyTorch 2.0+ (for U-Net edge detection; optional — Canny fallback available)

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

5. **(Optional) Train the U-Net edge detection model:**
   - Open `notebooks/HoloPaint_Model_Training.ipynb` in Google Colab
   - Run all cells to train on the BSDS500 dataset
   - Download the exported `holopaint_unet.pth` file to the project root
   - Without this file, HoloPaint will use Canny edge detection as a fallback

## Usage

Run the application:
```bash
python main.py
```

A menu screen appears with two modes:
- **Press 1** → HoloDraw (freehand drawing)
- **Press 2** → HoloPaint (AI edge detection + painting)

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera INDEX` | 0 | Webcam device index |
| `--width WIDTH` | 1280 | Capture width in pixels |
| `--height HEIGHT` | 720 | Capture height in pixels |
| `--landmarks` | Off | Show hand landmark overlay for debugging |

## Gesture Guide

| Gesture | How To | Action |
|---------|--------|--------|
| **Index finger only** | Extend only the index finger | Draw on the canvas |
| **Index + Middle** | Extend index and middle fingers | Navigate (move cursor) |
| **Pinch** | Touch thumb tip to index tip | Select tool / Flood fill |
| **Index + Middle + Ring** | Extend three fingers (curled pinky) | **Zoom & Pan mode** |
| **Fist** | Close all fingers | Undo |
| **Thumb up** | Extend only the thumb | Redo |
| **Open palm** | Extend all five fingers | Idle / Stop |

## HoloPaint Keyboard Controls

| Key | Action |
|-----|--------|
| `R` | Fetch a random image from the internet |
| `U` | Upload an image from your computer |
| `+/-` | Adjust edge detection threshold |
| `Enter` | Confirm threshold and start painting |
| `N` | Load a new image |
| `T` | Return to threshold adjustment |
| `S` | Save the painted result |
| `Q` | Quit |

## Project Structure

```
HoloDraw/
├── main.py                     # Application entry point & mode routing
├── config.py                   # Constants, enums, and settings
├── menu_screen.py              # Mode selection screen (HoloDraw / HoloPaint)
├── app_controller.py           # HoloDraw mode controller
├── holopaint_controller.py     # HoloPaint mode controller
├── hand_tracker.py             # MediaPipe hand landmark detection
├── gesture_recognizer.py       # Gesture classification from landmarks
├── drawing_engine.py           # Canvas management and undo/redo
├── shape_tools.py              # Shape drawing implementations
├── ui_overlay.py               # Toolbar and status bar rendering
├── edge_detector.py            # U-Net model + edge detection inference
├── image_fetcher.py            # Random image fetching (Lorem Picsum)
├── hand_landmarker.task        # MediaPipe HandLandmarker model (float16)
├── holopaint_unet.pth          # Trained U-Net weights (after training)
├── notebooks/
│   ├── HoloPaint_Model_Training.ipynb   # U-Net training pipeline (Colab)
│   └── HoloPaint_Model_Evaluation.ipynb # Model evaluation & visualizations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Architecture

### HoloDraw Pipeline
1. **HandTracker** captures frames and detects 21 hand landmarks using MediaPipe.
2. **GestureRecognizer** classifies the hand pose into a gesture using geometric rules.
3. **AppController** maps gestures to drawing actions and manages state.
4. **DrawingEngine** maintains the canvas and undo/redo history.
5. **UIOverlay** renders the toolbar and status bar.

### HoloPaint Pipeline
1. **ImageFetcher** acquires images (random fetch or file upload).
2. **EdgeDetector** runs the trained U-Net model to extract edge contours.
3. **HoloPaintController** manages the 3-state workflow (select → threshold → paint).
4. Reuses **HandTracker**, **GestureRecognizer**, and **DrawingEngine** for painting.

### U-Net Edge Detection Model
The U-Net is a fully convolutional encoder-decoder network with skip connections:
- **Encoder**: 4 stages (3→64→128→256→512 channels) with MaxPool downsampling
- **Bottleneck**: 1024-channel feature extraction at 16× reduced resolution
- **Decoder**: 4 stages with transposed convolution upsampling + skip concatenation
- **Output**: 1×1 convolution → Sigmoid for edge probability map
- **Training**: BSDS500 dataset, Weighted BCE loss, Adam optimizer, 50 epochs

## Troubleshooting

- **Hand not detected**: Ensure adequate, even lighting.
- **Erratic gesture switching**: Make deliberate gestures. The system includes debouncing.
- **Low FPS**: Try `--width 640 --height 480`.
- **Camera not opening**: Check that no other app is using the webcam.
- **Edge detection slow**: The U-Net runs on CPU. First inference (including tiled processing) may take 200-400ms on Apple M4 Pro hardware.
- **No trained model**: HoloPaint will use Canny edge detection as fallback.

## License

This project is provided for educational and personal use.
