"""
HoloPaint Edge Detector Module
---------------------------------
Provides the U-Net deep learning model for edge/contour detection and
an inference wrapper that processes images into paintable outline maps.

Architecture:
    The U-Net consists of a 4-stage encoder (downsampling path), a bottleneck,
    and a 4-stage decoder (upsampling path) with skip connections. Each stage
    uses two 3×3 convolutions with BatchNorm and ReLU activation.

    Input:  RGB image tensor  (B, 3, 256, 256)
    Output: Edge probability  (B, 1, 256, 256)  ∈ [0, 1]

Fallback:
    If PyTorch or the trained model is unavailable, the module falls back
    to OpenCV's Canny edge detector for basic outline extraction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

# Attempt to import PyTorch — fall back gracefully if unavailable
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# U-Net Architecture
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    import torch.nn.functional as F

    class DoubleConv(nn.Module):
        """
        Double convolution block used in the DexiNed encoder.
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class DexiNed(nn.Module):
        """
        DexiNed (Dense Extreme Inception Network) for edge/contour detection.

        This architecture utilizes deep supervision to produce multiple edge
        maps at different scales, and then fuses them. It natively generates
        thin, "coloring book" style outlines without requiring NMS.

        Args:
            in_channels:  Number of input image channels (default: 3 for RGB).
        """

        def __init__(self, in_channels: int = 3):
            super().__init__()

            # --- Backbone (Encoder) ---
            self.enc1 = DoubleConv(in_channels, 32)
            self.enc2 = DoubleConv(32, 64)
            self.enc3 = DoubleConv(64, 128)
            self.enc4 = DoubleConv(128, 256)
            self.enc5 = DoubleConv(256, 512)
            self.enc6 = DoubleConv(512, 512)
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # --- Deep Supervision Side Outputs ---
            self.side1 = nn.Conv2d(32, 1, kernel_size=1)
            self.side2 = nn.Conv2d(64, 1, kernel_size=1)
            self.side3 = nn.Conv2d(128, 1, kernel_size=1)
            self.side4 = nn.Conv2d(256, 1, kernel_size=1)
            self.side5 = nn.Conv2d(512, 1, kernel_size=1)
            self.side6 = nn.Conv2d(512, 1, kernel_size=1)

            # --- Fusion Layer ---
            self.fuse = nn.Conv2d(6, 1, kernel_size=1)

        def forward(self, x: torch.Tensor):
            """
            Forward pass through DexiNed.

            Returns:
                If training: List of 6 side outputs + 1 fused output (raw logits).
                If eval: The fused edge probability map of shape (B, 1, H, W).
            """
            h, w = x.shape[2:]

            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            e5 = self.enc5(self.pool(e4))
            e6 = self.enc6(self.pool(e5))

            # Side outputs (deep supervision)
            s1 = self.side1(e1)
            s2 = F.interpolate(self.side2(e2), size=(h, w), mode='bilinear', align_corners=False)
            s3 = F.interpolate(self.side3(e3), size=(h, w), mode='bilinear', align_corners=False)
            s4 = F.interpolate(self.side4(e4), size=(h, w), mode='bilinear', align_corners=False)
            s5 = F.interpolate(self.side5(e5), size=(h, w), mode='bilinear', align_corners=False)
            s6 = F.interpolate(self.side6(e6), size=(h, w), mode='bilinear', align_corners=False)

            # Fusion
            fused = self.fuse(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))

            if self.training:
                return [s1, s2, s3, s4, s5, s6, fused]
            else:
                return torch.sigmoid(fused)


    class UNetEdgeDetector(nn.Module):
        """
        U-Net for Edge Detection (original trained model).

        Full encoder-decoder with skip connections. Produces significantly
        better edge maps than the lightweight DexiNed encoder-only model.

        Architecture:
            Encoder: 3 → 64 → 128 → 256 → 512
            Bottleneck: 1024 channels
            Decoder: 1024 → 512 → 256 → 128 → 64 → 1 (with skip connections)
        """
        def __init__(self, in_channels=3, out_channels=1):
            super().__init__()
            # Encoder
            self.enc1 = DoubleConv(in_channels, 64)
            self.enc2 = DoubleConv(64, 128)
            self.enc3 = DoubleConv(128, 256)
            self.enc4 = DoubleConv(256, 512)
            self.pool = nn.MaxPool2d(2, 2)

            # Bottleneck
            self.bottleneck = DoubleConv(512, 1024)

            # Decoder
            self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(128, 64)

            # Output
            self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        def forward(self, x):
            # Encoder path
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            # Bottleneck
            b = self.bottleneck(self.pool(e4))

            # Decoder path with skip connections
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return torch.sigmoid(self.output_conv(d1))


# ---------------------------------------------------------------------------
# Edge Detector Inference Wrapper
# ---------------------------------------------------------------------------

class EdgeDetector:
    """
    High-level interface for edge detection on images.

    Attempts to load models in priority order:
        1. U-Net (holopaint_unet.pth) — full encoder-decoder, best quality
        2. DexiNed (holopaint_dexined.pth) — encoder-only, lighter
        3. Canny fallback — no deep learning required

    Attributes:
        model: The loaded model (or None if using Canny fallback).
        device: PyTorch device (CPU).
        use_canny: Whether the Canny fallback is active.
        model_type: 'unet', 'dexined', or 'canny'.
    """

    # Default model input resolution
    INPUT_SIZE = (256, 256)

    def __init__(self, model_path: str = "holopaint_unet.pth"):
        """
        Initialize the edge detector.

        Tries U-Net first, then DexiNed, then Canny fallback.

        Args:
            model_path: Path to the trained model weights (.pth file).
        """
        self.model = None
        self.device = None
        self.use_canny = False
        self.model_type = "canny"

        if not TORCH_AVAILABLE:
            print("[EdgeDetector] PyTorch not available. Using Canny fallback.")
            self.use_canny = True
            return

        self.device = torch.device("cpu")

        # --- Priority 1: Try U-Net (best quality, 124MB) ---
        if self._try_load_unet(model_path):
            return
        # If a non-unet path was given, also try the default unet path
        if model_path != "holopaint_unet.pth":
            if self._try_load_unet("holopaint_unet.pth"):
                return

        # --- Priority 2: Try DexiNed (lighter, 37MB) ---
        for dex_path in ["holopaint_dexined.pth", "best_dexined.pth"]:
            if self._try_load_dexined(dex_path):
                return

        # --- Priority 3: Canny fallback ---
        print("[EdgeDetector] No model loaded. Using Canny fallback.")
        self.use_canny = True

    def _try_load_unet(self, path: str) -> bool:
        """Attempt to load U-Net model from the given path."""
        try:
            model = UNetEdgeDetector(in_channels=3)
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.model_type = "unet"
            print(f"[EdgeDetector] U-Net model loaded from: {path}")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"[EdgeDetector] U-Net load failed ({path}): {e}")
            return False

    def _try_load_dexined(self, path: str) -> bool:
        """Attempt to load DexiNed model from the given path."""
        try:
            model = DexiNed(in_channels=3)
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.model_type = "dexined"
            print(f"[EdgeDetector] DexiNed model loaded from: {path}")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"[EdgeDetector] DexiNed load failed ({path}): {e}")
            return False

    def detect_edges(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Extract edges from an image and return a paintable outline map.

        Args:
            image: Input image in BGR format (OpenCV convention).
            threshold: Edge sensitivity threshold (0.0–1.0).
                       Lower values produce more edges; higher values produce
                       fewer, stronger edges. Only used with the U-Net model.
                       For Canny fallback, this scales the Canny thresholds.

        Returns:
            A binary edge map as a BGR image (white background, black edges)
            at the same resolution as the input image.
        """
        if self.use_canny:
            return self._detect_canny(image, threshold)
        elif self.model_type == "unet":
            return self._detect_unet(image, threshold)
        else:
            return self._detect_dexined(image, threshold)
    def _detect_unet(self, image: np.ndarray, threshold: float) -> np.ndarray:
        """
        Run edge detection using the U-Net model with tiled inference, TTA,
        and hybrid Canny fusion for maximum edge quality.

        Pipeline:
            1. Tiled inference — process overlapping 256×256 patches at full
               resolution instead of downscaling the whole image.
            2. Test-time augmentation — average with horizontally flipped prediction.
            3. Hybrid Canny fusion — catch fine details the model misses.
            4. Post-processing — hysteresis threshold, close gaps, remove noise.
        """
        original_h, original_w = image.shape[:2]

        # --- Step 1 + 2: Tiled inference with TTA ---
        edge_map_full = self._tiled_inference(image)

        # --- Step 3: Hybrid Canny fusion ---
        # Run a lightweight Canny pass to catch thin edges the model misses.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        canny_low = int(30 + threshold * 70)
        canny_high = int(80 + threshold * 120)
        canny_edges = cv2.Canny(blurred, canny_low, canny_high)

        # Normalize Canny to [0, 1] and weight it lower than the model
        canny_norm = canny_edges.astype(np.float32) / 255.0
        # Fuse: model output dominates, Canny fills in gaps
        fused = np.clip(edge_map_full + canny_norm * 0.3, 0, 1)

        # --- Step 4: Hysteresis thresholding ---
        high_thresh = threshold
        low_thresh = threshold * 0.35

        strong = (fused >= high_thresh).astype(np.uint8) * 255
        weak = (fused >= low_thresh).astype(np.uint8) * 255

        connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        strong_dilated = cv2.dilate(strong, connect_kernel, iterations=1)
        binary = cv2.bitwise_or(strong, cv2.bitwise_and(weak, strong_dilated))

        # Morphological closing to bridge micro-gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

        # Bridge endpoint gaps
        binary = self._bridge_endpoint_gaps(binary, search_radius=14)

        # Noise removal — keep only significant contours
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = max(25, int((original_w * original_h) * 0.0003))
        clean_binary = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_binary[labels == i] = 255

        # Solidify lines
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clean_binary = cv2.dilate(clean_binary, dilate_kernel, iterations=1)

        # Invert: black edges on white background
        outline = 255 - clean_binary
        outline_bgr = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(outline_bgr, (0, 0), (original_w - 1, original_h - 1), (0, 0, 0), 2)

        return outline_bgr

    def _run_model_on_patch(self, patch_bgr: np.ndarray) -> np.ndarray:
        """
        Run the model on a single 256×256 BGR patch.

        Includes test-time augmentation (horizontal flip averaging).

        Args:
            patch_bgr: A 256×256 BGR image patch.

        Returns:
            Edge probability map of shape (256, 256) in [0, 1].
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - mean) / std

        # Original prediction
        with torch.no_grad():
            pred = self.model(tensor.unsqueeze(0)).squeeze().numpy()

        # TTA: horizontal flip and average
        tensor_flip = torch.flip(tensor, dims=[2])
        with torch.no_grad():
            pred_flip = self.model(tensor_flip.unsqueeze(0)).squeeze().numpy()
        pred_flip = np.flip(pred_flip, axis=1)

        return (pred + pred_flip) / 2.0

    def _tiled_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-scale inference combining a global pass with targeted tiles.

        Strategy (much faster than exhaustive tiling):
            1. Global pass at 256×256 — captures overall structure (with TTA).
            2. Tile pass at full resolution — adds fine detail (no TTA, larger stride).
            3. Weighted blend of both scales.

        Args:
            image: Input BGR image at any resolution.

        Returns:
            Edge probability map at the original resolution, values in [0, 1].
        """
        h, w = image.shape[:2]
        tile_size = self.INPUT_SIZE[0]  # 256

        # For small images, just resize and run once with TTA
        if h <= tile_size and w <= tile_size:
            resized = cv2.resize(image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            pred = self._run_model_on_patch(resized)
            return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- Scale 1: Global pass (overall structure) ---
        global_resized = cv2.resize(image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        global_pred = self._run_model_on_patch(global_resized)  # with TTA
        global_full = cv2.resize(global_pred, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- Scale 2: Tile pass (fine local detail, no TTA for speed) ---
        stride = tile_size * 3 // 4  # 192 — 25% overlap (faster than 50%)

        pad_h = (stride - (h % stride)) % stride
        pad_w = (stride - (w % stride)) % stride
        padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        ph, pw = padded.shape[:2]

        acc = np.zeros((ph, pw), dtype=np.float64)
        weight_acc = np.zeros((ph, pw), dtype=np.float64)

        # Soft blending window to avoid tile seam artifacts
        window_1d = np.hanning(tile_size).astype(np.float64)
        window_2d = np.outer(window_1d, window_1d)
        window_2d = np.clip(window_2d, 0.01, 1.0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tile_count = 0
        for y in range(0, ph - tile_size + 1, stride):
            for x in range(0, pw - tile_size + 1, stride):
                tile = padded[y:y + tile_size, x:x + tile_size]

                # Direct inference (no TTA for speed)
                rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensor = (tensor - mean) / std
                with torch.no_grad():
                    pred = self.model(tensor.unsqueeze(0)).squeeze().numpy()

                acc[y:y + tile_size, x:x + tile_size] += pred * window_2d
                weight_acc[y:y + tile_size, x:x + tile_size] += window_2d
                tile_count += 1

        weight_acc = np.maximum(weight_acc, 1e-8)
        tile_full = (acc / weight_acc)[:h, :w].astype(np.float32)

        print(f"[EdgeDetector] Multi-scale: 1 global + {tile_count} tiles processed")

        # --- Blend scales: tiles dominate (fine detail), global fills gaps ---
        blended = np.maximum(tile_full * 0.7 + global_full * 0.3, 
                             np.maximum(tile_full, global_full * 0.5))

        return np.clip(blended, 0, 1).astype(np.float32)

    def _detect_dexined(self, image: np.ndarray, threshold: float) -> np.ndarray:
        """Run edge detection using the trained DexiNed model."""
        original_h, original_w = image.shape[:2]

        # Preprocess: resize, convert BGR→RGB, normalize to [0, 1], add batch dim
        resized = cv2.resize(image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        # Normalize with ImageNet statistics (matches training preprocessing)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        tensor = tensor.unsqueeze(0)  # (1, 3, 256, 256)

        # Inference
        with torch.no_grad():
            output = self.model(tensor)  # (1, 1, 256, 256)

        # Postprocess
        edge_map = output.squeeze().numpy()  # (256, 256) in [0, 1]

        # --- FIX 1: Resize the continuous probability map BEFORE thresholding ---
        # Using INTER_LINEAR (bilinear) on the soft probability map preserves
        # sub-pixel edge positions that INTER_NEAREST on a binary map would drop.
        # This is the primary cause of micro-disconnections at thin junctions.
        edge_map_full = cv2.resize(
            edge_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )

        # --- FIX 2: Hysteresis thresholding instead of hard binary cutoff ---
        # A single hard threshold discards weak-but-connected pixels at corners
        # and junctions where model confidence is lower. Hysteresis keeps weak
        # edge pixels (>= low_thresh) as long as they are connected to strong
        # edge pixels (>= threshold), producing continuous contours.
        high_thresh = threshold
        low_thresh = threshold * 0.4  # Keep weak edges connected to strong ones

        strong = (edge_map_full >= high_thresh).astype(np.uint8) * 255
        weak = (edge_map_full >= low_thresh).astype(np.uint8) * 255

        # Dilate strong edges slightly so that nearby weak pixels connect to them
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        strong_dilated = cv2.dilate(strong, connect_kernel, iterations=1)

        # Keep weak pixels only where they touch a strong region
        binary = cv2.bitwise_and(weak, strong_dilated)
        # Always keep the original strong pixels
        binary = cv2.bitwise_or(binary, strong)

        # --- 1. Gap Closing ("Colorable") ---
        # Use a larger elliptical kernel to bridge micro-gaps that survive
        # the improved thresholding. 5×5 reliably closes 2-3 px gaps.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

        # --- FIX 3: Skeleton-aware endpoint gap bridging ---
        # After closing, some gaps may still remain (especially at T-junctions).
        # Skeletonize the edges, find endpoints, and draw short line segments
        # to the nearest other endpoint within a search radius.
        binary = self._bridge_endpoint_gaps(binary, search_radius=12)

        # --- 2. Noise Removal ---
        # Use connected components to filter out small isolated dots and texture noise
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Minimum area to keep a line (e.g., 0.05% of total pixels, or at least 20 pixels)
        min_area = max(20, int((original_w * original_h) * 0.0002))

        clean_binary = np.zeros_like(binary)
        for i in range(1, num_labels):  # skip background 0
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_binary[labels == i] = 255

        # --- 3. Solidify Lines ---
        # Dilate slightly to ensure the lines are perfectly solid and impermeable to the fill bucket.
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clean_binary = cv2.dilate(clean_binary, dilate_kernel, iterations=1)

        # Invert: black edges on white background (paintable format)
        outline = 255 - clean_binary

        # Convert to 3-channel BGR for canvas compatibility
        outline_bgr = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)

        # Add a solid black border to prevent flood fill from leaking out of the canvas
        cv2.rectangle(outline_bgr, (0, 0), (original_w - 1, original_h - 1), (0, 0, 0), 2)

        return outline_bgr

    @staticmethod
    def _bridge_endpoint_gaps(binary: np.ndarray, search_radius: int = 12) -> np.ndarray:
        """
        Find skeleton endpoints and bridge nearby ones to close small gaps.

        This handles micro-disconnections that morphological closing alone cannot
        fix (e.g., at T-junctions or where two contour fragments almost meet).

        Algorithm:
            1. Thin the edge map to a 1-pixel skeleton.
            2. Detect endpoints via vectorized 3×3 neighbor counting.
            3. For each endpoint, find the nearest endpoint from a *different*
               connected component within `search_radius`.
            4. Draw a line segment connecting the two endpoints.

        Args:
            binary: Input binary edge map (uint8, 0 or 255).
            search_radius: Maximum pixel distance to search for a partner endpoint.

        Returns:
            Binary edge map with gaps bridged.
        """
        # Skeletonize for accurate endpoint detection
        try:
            skeleton = cv2.ximgproc.thinning(binary)
        except (AttributeError, cv2.error):
            # cv2.ximgproc not available — fall back without thinning
            skeleton = binary.copy()

        # --- Vectorized endpoint detection ---
        # Count 8-connected neighbors for every pixel using a convolution kernel.
        # An endpoint is a skeleton pixel with exactly 1 skeleton neighbor.
        skel_01 = (skeleton > 0).astype(np.float32)
        neighbor_kernel = np.array(
            [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]], dtype=np.float32
        )
        neighbor_count = cv2.filter2D(skel_01, -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)

        # Endpoints: skeleton pixel AND exactly 1 neighbor
        endpoint_mask = (skel_01 > 0) & (neighbor_count >= 0.5) & (neighbor_count < 1.5)
        ey, ex = np.where(endpoint_mask)

        if len(ey) < 2:
            return binary

        # Convert to list of (x, y) tuples
        endpoints = list(zip(ex.tolist(), ey.tolist()))

        # Label connected components so we only bridge *different* components
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

        result = binary.copy()
        used = set()

        # Build arrays for vectorized distance computation
        ep_arr = np.array(endpoints, dtype=np.float64)  # shape (N, 2) — columns are x, y

        for i in range(len(endpoints)):
            if i in used:
                continue
            x1, y1 = endpoints[i]
            label1 = labels[y1, x1]

            # Compute distances from endpoint i to all others
            diffs = ep_arr - ep_arr[i]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])

            best_dist = search_radius + 1
            best_j = -1

            # Sort candidate indices by distance for efficient search
            candidates = np.argsort(dists)
            for j_idx in candidates:
                j = int(j_idx)
                if j <= i or j in used:
                    continue
                if dists[j] > search_radius:
                    break  # All remaining are farther
                x2, y2 = endpoints[j]
                label2 = labels[y2, x2]
                if label2 == label1:
                    continue  # Same component — no need to bridge
                best_dist = dists[j]
                best_j = j
                break  # Found the nearest from a different component

            if best_j >= 0:
                x2, y2 = endpoints[best_j]
                cv2.line(result, (x1, y1), (x2, y2), 255, 1, cv2.LINE_AA)
                used.add(i)
                used.add(best_j)

        return result

    def _detect_canny(self, image: np.ndarray, threshold: float) -> np.ndarray:
        """
        Advanced edge detection pipeline optimized for clean, paintable outlines.

        Combines multiple techniques to produce coloring-book-style contours:
            1. Bilateral filtering — smooths textures while preserving edges
            2. Multi-scale Canny — captures both coarse structure and fine detail
            3. Adaptive thresholding — extracts edges based on local contrast
            4. Contour extraction — draws clean, continuous boundary lines
            5. Morphological cleanup — connects fragments and removes noise

        The threshold parameter controls the detail level:
            Low (0.05–0.2)  → more edges, finer detail
            Mid (0.2–0.5)   → balanced, good for most images
            High (0.5–1.0)  → fewer edges, only major outlines

        Args:
            image: Input BGR image.
            threshold: Detail level control (0.0–1.0).

        Returns:
            Clean black-on-white outline image (BGR).
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- Stage 1: Edge-preserving smoothing ---
        # Bilateral filter removes texture noise while keeping sharp edges.
        # Stronger filtering at higher thresholds (fewer details).
        d = int(7 + threshold * 8)           # Filter diameter: 7–15
        sigma_color = 50 + threshold * 100   # Color sigma: 50–150
        sigma_space = 50 + threshold * 100   # Spatial sigma: 50–150
        smoothed = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

        # --- Stage 2: Multi-scale Canny edge detection ---
        # Coarse scale: captures major structural outlines
        coarse_blur = cv2.GaussianBlur(smoothed, (7, 7), 2.0)
        coarse_low = int(20 + threshold * 60)
        coarse_high = int(60 + threshold * 140)
        coarse_edges = cv2.Canny(coarse_blur, coarse_low, coarse_high)

        # Fine scale: captures important detail edges
        fine_blur = cv2.GaussianBlur(smoothed, (3, 3), 0.8)
        fine_low = int(40 + threshold * 80)
        fine_high = int(100 + threshold * 155)
        fine_edges = cv2.Canny(fine_blur, fine_low, fine_high)

        # Combine: always keep coarse edges, add fine edges weighted by threshold
        # At low threshold (high detail), include more fine edges
        combined = cv2.bitwise_or(coarse_edges, fine_edges)

        # --- Stage 3: Adaptive thresholding for local contrast edges ---
        # Captures edges that Canny might miss (e.g. soft gradients)
        block_size = int(11 + threshold * 20) | 1  # Must be odd: 11–31
        c_val = int(2 + threshold * 8)              # Constant: 2–10
        adaptive = cv2.adaptiveThreshold(
            smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_val,
        )

        # Thin the adaptive threshold result to single-pixel edges
        try:
            adaptive_thin = cv2.ximgproc.thinning(adaptive)
        except (AttributeError, cv2.error):
            adaptive_thin = adaptive

        # Blend adaptive edges with Canny (adaptive fills in gaps)
        # Weight adaptive contribution inversely with threshold
        adaptive_weight = max(0.0, 0.5 - threshold * 0.6)
        if adaptive_weight > 0.05:
            combined = cv2.bitwise_or(
                combined,
                cv2.threshold(
                    (adaptive_thin * adaptive_weight).astype(np.uint8),
                    30, 255, cv2.THRESH_BINARY
                )[1],
            )

        # --- Stage 4: Contour-based clean edges ---
        # Find contours and redraw them for cleaner, smoother lines
        contours, hierarchy = cv2.findContours(
            combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Filter contours: remove tiny noise fragments
        min_contour_area = max(10, int((h * w) * 0.0001 * (1 + threshold * 5)))
        min_contour_len = max(15, int(min(h, w) * 0.02 * (1 + threshold * 3)))

        contour_canvas = np.zeros_like(gray)
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, closed=False)

            if area >= min_contour_area or length >= min_contour_len:
                # Simplify contour slightly for smoother lines
                epsilon = 0.5 + threshold * 1.5  # Approximation accuracy
                approx = cv2.approxPolyDP(contour, epsilon, closed=False)
                cv2.drawContours(contour_canvas, [approx], -1, 255, 1, cv2.LINE_AA)

        # --- Stage 5: Morphological cleanup ---
        # Close gaps in contour lines to prevent flood fill leaks.
        # Higher threshold = fewer edges = we need a larger kernel to bridge gaps.
        close_size = int(3 + threshold * 6)  # Kernel size: 3 to 9
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        contour_canvas = cv2.morphologyEx(contour_canvas, cv2.MORPH_CLOSE, close_kernel)

        # Remove isolated noise pixels
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        contour_canvas = cv2.morphologyEx(contour_canvas, cv2.MORPH_OPEN, open_kernel)

        # Dilate to ensure solid, unbroken boundaries for the fill tool
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        contour_canvas = cv2.dilate(contour_canvas, dilate_kernel, iterations=1)

        # --- Final output ---
        # Invert: black edges on white background (paintable format)
        outline = 255 - contour_canvas

        # Convert to 3-channel BGR for canvas compatibility
        outline_bgr = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)

        # Add a solid black border to prevent flood fill from leaking out of the canvas
        cv2.rectangle(outline_bgr, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)

        return outline_bgr
