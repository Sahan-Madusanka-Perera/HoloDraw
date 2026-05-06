# %% [markdown]
# # HoloPaint U-Net Edge Detection Model — Training Pipeline
#
# This notebook trains a **U-Net convolutional neural network** for edge/contour
# detection on the **BSDS500** dataset. The trained model is exported as
# `holopaint_unet.pth` for use in the HoloPaint application.
#
# ## Architecture Overview
# - **Encoder**: 4 downsampling blocks (Conv-BN-ReLU × 2 + MaxPool)
# - **Bottleneck**: 1024-channel feature extraction
# - **Decoder**: 4 upsampling blocks with skip connections
# - **Output**: 1-channel sigmoid edge probability map
#
# ## Key Deep Learning Concepts Demonstrated
# 1. Convolutional Neural Networks (CNNs)
# 2. Encoder-Decoder architecture
# 3. Skip connections for spatial detail recovery
# 4. Batch Normalization for training stability
# 5. Weighted Binary Cross-Entropy for class imbalance
# 6. Data augmentation (flips, rotations, crops)
# 7. Learning rate scheduling
# 8. Model evaluation with F1-score

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
# Install dependencies (Colab)
# !pip install torch torchvision matplotlib scikit-image

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Download BSDS500 Dataset

# %%
# Download BSDS500 dataset
import urllib.request
import tarfile

DATASET_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
DATASET_DIR = Path("./data")
BSR_DIR = DATASET_DIR / "BSR" / "BSDS500" / "data"

if not BSR_DIR.exists():
    print("Downloading BSDS500 dataset...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATASET_DIR / "BSR_bsds500.tgz"

    urllib.request.urlretrieve(DATASET_URL, archive_path)
    print("Extracting...")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=DATASET_DIR)

    archive_path.unlink()
    print("Dataset ready!")
else:
    print(f"Dataset already exists at {BSR_DIR}")

# %% [markdown]
# ## 3. Dataset Class with Augmentation

# %%
import scipy.io

class BSDS500Dataset(Dataset):
    """
    BSDS500 edge detection dataset.

    Loads RGB images and their corresponding human-annotated edge ground truth
    maps. Multiple annotators' edge maps are averaged to create a consensus
    edge probability map, then binarized.

    Augmentation:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (±15°)
        - Random crop to 256×256
    """

    def __init__(self, split: str = "train", img_size: int = 256, augment: bool = True):
        """
        Args:
            split: One of 'train', 'val', or 'test'.
            img_size: Target image size (square crop).
            augment: Whether to apply data augmentation.
        """
        self.img_size = img_size
        self.augment = augment

        img_dir = BSR_DIR / "images" / split
        gt_dir = BSR_DIR / "groundTruth" / split

        self.image_paths = sorted(img_dir.glob("*.jpg"))
        self.gt_paths = sorted(gt_dir.glob("*.mat"))

        assert len(self.image_paths) == len(self.gt_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.gt_paths)} ground truths"

        print(f"[{split}] Loaded {len(self.image_paths)} image-GT pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Load ground truth edge map from .mat file
        mat = scipy.io.loadmat(self.gt_paths[idx])
        gt_data = mat["groundTruth"][0]

        # Average all annotators' edge maps
        edge_maps = []
        for i in range(len(gt_data)):
            boundaries = gt_data[i][0][0][1]  # 'Boundaries' field
            edge_maps.append(boundaries.astype(np.float32))

        avg_edges = np.mean(edge_maps, axis=0)
        # Binarize: pixel is edge if >50% of annotators agree
        binary_edges = (avg_edges > 0.5).astype(np.float32)
        edge_pil = Image.fromarray((binary_edges * 255).astype(np.uint8), mode="L")

        # Apply synchronized transforms
        image, edge_pil = self._transform(image, edge_pil)

        return image, edge_pil

    def _transform(self, image, edge):
        """Apply synchronized augmentation to image and edge map."""
        # Resize to slightly larger than crop size
        resize_to = self.img_size + 32 if self.augment else self.img_size
        image = TF.resize(image, [resize_to, resize_to])
        edge = TF.resize(edge, [resize_to, resize_to], interpolation=T.InterpolationMode.NEAREST)

        if self.augment:
            # Random crop
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image = TF.crop(image, i, j, h, w)
            edge = TF.crop(edge, i, j, h, w)

            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                edge = TF.hflip(edge)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                edge = TF.vflip(edge)

            # Random rotation (±15°)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                edge = TF.rotate(edge, angle)
        else:
            # Center crop for validation/test
            image = TF.center_crop(image, [self.img_size, self.img_size])
            edge = TF.center_crop(edge, [self.img_size, self.img_size])

        # To tensor + normalize image
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Edge map to tensor (keep as binary 0/1)
        edge = TF.to_tensor(edge)
        edge = (edge > 0.5).float()

        return image, edge

# %%
# Create dataloaders
train_dataset = BSDS500Dataset(split="train", augment=True)
val_dataset = BSDS500Dataset(split="val", augment=False)
test_dataset = BSDS500Dataset(split="test", augment=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# %%
# Visualize a few training samples
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    img, edge = train_dataset[i]
    # Denormalize image for display
    img_display = img.permute(1, 2, 0).numpy()
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)

    axes[0, i].imshow(img_display)
    axes[0, i].set_title(f"Image {i+1}")
    axes[0, i].axis("off")

    axes[1, i].imshow(edge.squeeze(), cmap="gray")
    axes[1, i].set_title(f"Edge GT {i+1}")
    axes[1, i].axis("off")

plt.suptitle("BSDS500 Training Samples", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. U-Net Model Architecture

# %%
class ConvBlock(nn.Module):
    """
    Double convolution block: Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU

    This is the fundamental building block used in every encoder and decoder
    stage of the U-Net. Two successive convolutions allow the network to
    learn increasingly complex feature representations.
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

    def forward(self, x):
        return self.block(x)


class UNetEdgeDetector(nn.Module):
    """
    U-Net for Edge Detection

    Architecture:
        Encoder (Contracting Path):
            4 stages of ConvBlock + MaxPool(2×2)
            Channel progression: 3 → 64 → 128 → 256 → 512

        Bottleneck:
            ConvBlock at 1024 channels (deepest feature representation)

        Decoder (Expanding Path):
            4 stages of TransposedConv(2×2) + Concatenate(skip) + ConvBlock
            Channel progression: 1024 → 512 → 256 → 128 → 64

        Output:
            1×1 convolution → Sigmoid (edge probability map)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

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

# %%
# Instantiate model and print summary
model = UNetEdgeDetector().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Verify forward pass
dummy_input = torch.randn(1, 3, 256, 256).to(device)
dummy_output = model(dummy_input)
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {dummy_output.shape}")
assert dummy_output.shape == (1, 1, 256, 256), "Output shape mismatch!"
print("✓ Architecture verified!")

# %% [markdown]
# ## 5. Loss Function — Weighted Binary Cross-Entropy
#
# Edge pixels are **extremely sparse** (~5% of all pixels). Standard BCE
# would bias the model toward predicting "no edge" everywhere. We use
# **pos_weight** to up-weight edge pixels, forcing the model to pay
# attention to the minority class.

# %%
# Calculate class imbalance ratio from training set
edge_pixels = 0
total_pixels = 0
for i in range(min(50, len(train_dataset))):
    _, edge = train_dataset[i]
    edge_pixels += edge.sum().item()
    total_pixels += edge.numel()

edge_ratio = edge_pixels / total_pixels
pos_weight = (1 - edge_ratio) / edge_ratio
print(f"Edge pixel ratio: {edge_ratio:.4f} ({edge_ratio*100:.1f}%)")
print(f"Positive weight:  {pos_weight:.1f}")

# Cap the weight to avoid instability
pos_weight_tensor = torch.tensor([min(pos_weight, 15.0)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
print(f"Using pos_weight = {pos_weight_tensor.item():.1f}")

# Note: We'll use BCEWithLogitsLoss (raw logits) instead of BCELoss
# for numerical stability. This means we remove the sigmoid from the
# forward pass during training and add it back during inference.

# %% [markdown]
# ## 6. Training Loop

# %%
# Modify model for BCEWithLogitsLoss (remove sigmoid during training)
class UNetTraining(UNetEdgeDetector):
    """U-Net variant that outputs raw logits (no sigmoid) for BCEWithLogitsLoss."""
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output_conv(d1)  # Raw logits, no sigmoid

model = UNetTraining().to(device)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Training configuration
NUM_EPOCHS = 50
GRAD_CLIP = 1.0

# Logging
train_losses = []
val_losses = []
val_f1_scores = []
best_val_loss = float('inf')

# %%
print("=" * 60)
print("Starting Training")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    # --- Training phase ---
    model.train()
    epoch_loss = 0.0

    for batch_idx, (images, edges) in enumerate(train_loader):
        images = images.to(device)
        edges = edges.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, edges)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation phase ---
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, edges in val_loader:
            images = images.to(device)
            edges = edges.to(device)

            outputs = model(images)
            loss = criterion(outputs, edges)
            val_loss += loss.item()

            # Compute predictions for F1-score
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(edges.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # F1-score
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    val_f1_scores.append(f1)

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")

    # Print progress
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val F1: {f1:.4f} | "
          f"LR: {current_lr:.2e}")

print("\nTraining complete!")

# %% [markdown]
# ## 7. Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(train_losses, label="Train Loss", color="#2196F3", linewidth=2)
ax1.plot(val_losses, label="Val Loss", color="#FF5722", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# F1-score curve
ax2.plot(val_f1_scores, label="Val F1-Score", color="#4CAF50", linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("F1 Score")
ax2.set_title("Validation F1 Score")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("HoloPaint U-Net Training Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Export Model for HoloPaint

# %%
# Load the best model weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# Create the inference version (with sigmoid)
inference_model = UNetEdgeDetector().to(device)

# Copy weights from training model (same architecture, just different forward)
inference_state = model.state_dict()
inference_model.load_state_dict(inference_state)

# Save for HoloPaint app
export_path = "holopaint_unet.pth"
torch.save(inference_model.state_dict(), export_path)
file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
print(f"Model exported to: {export_path} ({file_size_mb:.1f} MB)")

# Save training logs for evaluation notebook
training_log = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_f1_scores": val_f1_scores,
    "best_val_loss": best_val_loss,
    "num_epochs": NUM_EPOCHS,
}
torch.save(training_log, "training_log.pth")
print("Training logs saved to: training_log.pth")

# %% [markdown]
# ## 9. Quick Inference Test

# %%
inference_model.eval()

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(4):
    img, gt = test_dataset[i]
    with torch.no_grad():
        pred = inference_model(img.unsqueeze(0).to(device))
    pred = pred.squeeze().cpu().numpy()

    # Denormalize image
    img_display = img.permute(1, 2, 0).numpy()
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)

    axes[0, i].imshow(img_display)
    axes[0, i].set_title("Input")
    axes[0, i].axis("off")

    axes[1, i].imshow(gt.squeeze(), cmap="gray")
    axes[1, i].set_title("Ground Truth")
    axes[1, i].axis("off")

    axes[2, i].imshow(pred, cmap="gray")
    axes[2, i].set_title("Predicted Edges")
    axes[2, i].axis("off")

plt.suptitle("U-Net Edge Detection — Test Set Predictions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("inference_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("Done! Copy 'holopaint_unet.pth' to your HoloDraw project root.")
