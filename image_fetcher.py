"""
HoloPaint Image Fetcher Module
---------------------------------
Fetches random images from the internet for use in HoloPaint mode.
Uses Lorem Picsum (https://picsum.photos) which requires no API key
and provides high-quality, royalty-free photographs.

The module handles network errors gracefully and provides a fallback
gradient image if no internet connection is available.
"""

import cv2
import numpy as np
import urllib.request
import urllib.error
from typing import Optional, List
from io import BytesIO


class ImageFetcher:
    """
    Fetches random images from Lorem Picsum for edge detection processing.

    Lorem Picsum provides random photographs at any requested resolution.
    No API key or authentication is required.

    Attributes:
        _cache: List of recently fetched images for quick re-use.
        _max_cache: Maximum number of images to keep in the cache.
    """

    # Lorem Picsum base URL
    BASE_URL = "https://picsum.photos"

    # Request timeout in seconds
    TIMEOUT = 10

    # User-Agent header to avoid being blocked
    USER_AGENT = "HoloPaint/1.0 (Educational Project)"

    def __init__(self, max_cache: int = 5):
        """
        Initialize the image fetcher.

        Args:
            max_cache: Maximum number of images to keep in the local cache.
        """
        self._cache: List[np.ndarray] = []
        self._max_cache = max_cache

    def fetch_random(
        self,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        """
        Fetch a random image from Lorem Picsum.

        Args:
            width: Desired image width in pixels.
            height: Desired image height in pixels.

        Returns:
            The fetched image as a BGR NumPy array (OpenCV format).
            If the fetch fails, returns a fallback gradient image.
        """
        url = f"{self.BASE_URL}/{width}/{height}"

        try:
            print(f"[ImageFetcher] Fetching random image from: {url}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": self.USER_AGENT},
            )
            response = urllib.request.urlopen(request, timeout=self.TIMEOUT)
            image_data = response.read()

            # Decode the image bytes into an OpenCV array
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                print("[ImageFetcher] Warning: Failed to decode image. Using fallback.")
                return self._create_fallback_image(width, height)

            # Add to cache
            self._add_to_cache(image)

            print(f"[ImageFetcher] Successfully fetched image: {image.shape[1]}x{image.shape[0]}")
            return image

        except urllib.error.URLError as e:
            print(f"[ImageFetcher] Network error: {e}. Using fallback image.")
            return self._get_cached_or_fallback(width, height)

        except Exception as e:
            print(f"[ImageFetcher] Unexpected error: {e}. Using fallback image.")
            return self._get_cached_or_fallback(width, height)

    def fetch_grayscale(
        self,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        """
        Fetch a random grayscale image from Lorem Picsum.

        Args:
            width: Desired image width in pixels.
            height: Desired image height in pixels.

        Returns:
            The fetched grayscale image as a BGR NumPy array.
        """
        url = f"{self.BASE_URL}/{width}/{height}?grayscale"

        try:
            print(f"[ImageFetcher] Fetching grayscale image from: {url}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": self.USER_AGENT},
            )
            response = urllib.request.urlopen(request, timeout=self.TIMEOUT)
            image_data = response.read()

            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return self._create_fallback_image(width, height)

            self._add_to_cache(image)
            return image

        except Exception as e:
            print(f"[ImageFetcher] Error fetching grayscale: {e}. Using fallback.")
            return self._get_cached_or_fallback(width, height)

    def load_from_file(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load an image from a local file path.

        Args:
            filepath: Path to the image file.

        Returns:
            The loaded image as a BGR NumPy array, or None if loading fails.
        """
        try:
            image = cv2.imread(filepath)
            if image is None:
                print(f"[ImageFetcher] Failed to load image: {filepath}")
                return None

            self._add_to_cache(image)
            print(f"[ImageFetcher] Loaded image from: {filepath} ({image.shape[1]}x{image.shape[0]})")
            return image

        except Exception as e:
            print(f"[ImageFetcher] Error loading file: {e}")
            return None

    def get_last_cached(self) -> Optional[np.ndarray]:
        """Return the most recently cached image, or None if cache is empty."""
        if self._cache:
            return self._cache[-1].copy()
        return None

    def _add_to_cache(self, image: np.ndarray) -> None:
        """Add an image to the cache, evicting the oldest if full."""
        self._cache.append(image.copy())
        if len(self._cache) > self._max_cache:
            self._cache.pop(0)

    def _get_cached_or_fallback(
        self,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Return a cached image if available, otherwise create a fallback."""
        if self._cache:
            print("[ImageFetcher] Using cached image.")
            return self._cache[-1].copy()
        return self._create_fallback_image(width, height)

    @staticmethod
    def _create_fallback_image(width: int, height: int) -> np.ndarray:
        """
        Create a simple gradient fallback image when no network is available.

        Generates a colorful gradient with geometric shapes so that edge
        detection still produces interesting outlines for painting.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            A BGR NumPy array containing the fallback image.
        """
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a two-tone gradient background
        for y in range(height):
            ratio = y / height
            # Blue-to-teal gradient
            image[y, :, 0] = int(180 * (1 - ratio) + 60 * ratio)   # Blue channel
            image[y, :, 1] = int(80 * (1 - ratio) + 180 * ratio)   # Green channel
            image[y, :, 2] = int(40 * (1 - ratio) + 100 * ratio)   # Red channel

        # Add some geometric shapes for interesting edge detection
        cx, cy = width // 2, height // 2

        # Large circle
        cv2.circle(image, (cx, cy), min(width, height) // 4, (255, 220, 180), -1, cv2.LINE_AA)

        # Rectangle
        cv2.rectangle(
            image,
            (width // 6, height // 6),
            (width // 3, height // 3),
            (200, 150, 250), -1,
        )

        # Triangle
        pts = np.array([
            [cx + width // 6, cy - height // 6],
            [cx + width // 3, cy + height // 6],
            [cx, cy + height // 6],
        ], np.int32)
        cv2.fillPoly(image, [pts], (180, 250, 200))

        # Star pattern
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            x_end = int(cx + (min(width, height) // 3) * np.cos(rad))
            y_end = int(cy + (min(width, height) // 3) * np.sin(rad))
            cv2.line(image, (cx, cy), (x_end, y_end), (255, 255, 220), 3, cv2.LINE_AA)

        # Add text label
        cv2.putText(
            image, "Offline Fallback", (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
        )

        return image
