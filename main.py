"""
HoloBrush - Hand Gesture Drawing Application
---------------------------------------------
Entry point for the HoloBrush application.

Usage:
    python main.py [--camera INDEX] [--width WIDTH] [--height HEIGHT]

This module parses command-line arguments, constructs the application
settings, and starts the main controller loop.
"""

import argparse
import sys

from config import AppSettings
from app_controller import AppController
from holopaint_controller import HoloPaintController

from menu_screen import MenuScreen


def parse_arguments() -> AppSettings:
    """
    Parse command-line arguments and return an AppSettings instance.

    Returns:
        Configured AppSettings with any command-line overrides applied.
    """
    parser = argparse.ArgumentParser(
        prog="HoloDraw",
        description="A hand gesture-based drawing application powered by MediaPipe.",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Capture resolution width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Capture resolution height in pixels (default: 720).",
    )
    parser.add_argument(
        "--landmarks", action="store_true",
        help="Show hand landmarks overlay for debugging.",
    )

    args = parser.parse_args()

    return AppSettings(
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        show_landmarks=args.landmarks,
    )


def main() -> None:
    """Application entry point."""
    print("=" * 60)
    print("  HoloDraw - Hand Gesture Drawing Application")
    print("=" * 60)
    print()

    settings = parse_arguments()
    menu = MenuScreen(settings.camera_width, settings.camera_height)
    mode = menu.run()

    if mode is None:
        return

    # Route to the appropriate controller based on the selected mode
    if mode == "HoloPaint":
        controller = HoloPaintController(settings)
    else:
        controller = AppController(settings, mode=mode)

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[HoloDraw] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
