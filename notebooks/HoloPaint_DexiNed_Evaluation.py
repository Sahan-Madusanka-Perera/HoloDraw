# %% [markdown]
# # HoloPaint DexiNed Model Evaluation
#
# This notebook evaluates the trained DexiNed edge detection model, calculates metrics,
# visualizes deep supervision intermediate outputs, and compares the final fused edge map
# against the ground truth.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import scipy.io
from sklearn.metrics import f1_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Load Trained Model

# %%
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )
    def forward(self, x): return self.block(x)

class DexiNed(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.enc5 = DoubleConv(256, 512)
        self.enc6 = DoubleConv(512, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.side1 = nn.Conv2d(32, 1, 1)
        self.side2 = nn.Conv2d(64, 1, 1)
        self.side3 = nn.Conv2d(128, 1, 1)
        self.side4 = nn.Conv2d(256, 1, 1)
        self.side5 = nn.Conv2d(512, 1, 1)
        self.side6 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))
        
        s1 = self.side1(e1)
        s2 = F.interpolate(self.side2(e2), size=(h, w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(e3), size=(h, w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(e4), size=(h, w), mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.side5(e5), size=(h, w), mode='bilinear', align_corners=False)
        s6 = F.interpolate(self.side6(e6), size=(h, w), mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))
        if self.training:
            return [s1, s2, s3, s4, s5, s6, fused]
        else:
            return torch.sigmoid(fused)

model = DexiNed().to(device)
try:
    model.load_state_dict(torch.load("holopaint_dexined.pth", map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("WARNING: holopaint_dexined.pth not found. Run training script first.")

# %% [markdown]
# ## 2. Training Curves

# %%
try:
    log = torch.load("dexined_training_log.pth", map_location="cpu", weights_only=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(log["train_losses"], label="Train")
    ax1.plot(log["val_losses"], label="Validation")
    ax1.set_title("DexiNed Deep Supervision Loss")
    ax1.legend(); ax1.grid(True)
    ax2.plot(log["val_f1_scores"], color="green")
    ax2.set_title("Validation F1 Score")
    ax2.grid(True)
    plt.savefig("dexined_curves.png")
    plt.show()
except FileNotFoundError:
    print("No training logs found.")

# %% [markdown]
# ## 3. Visual Comparison Grid

# %%
BIPED_DIR = Path("./data/BIPED/edges")
test_images = sorted((BIPED_DIR / "imgs" / "test" / "rgbr").glob("*.jpg"))
test_gts = sorted((BIPED_DIR / "edge_maps" / "test" / "rgbr").glob("*.png"))

if test_images:
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    cols = ["Original Image", "Ground Truth", "DexiNed Fused Edge"]
    for ax, col in zip(axes[0], cols): ax.set_title(col, fontweight="bold")
    
    for row in range(4):
        idx = row * 10
        if idx >= len(test_images): break
        img = Image.open(test_images[idx]).convert("RGB")
        img_resized = TF.resize(img, [256, 256])
        tensor = TF.normalize(TF.to_tensor(img_resized), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        with torch.no_grad():
            pred = model(tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            
        gt_pil = Image.open(test_gts[idx]).convert("L")
        gt_resized = TF.resize(gt_pil, [256, 256], interpolation=T.InterpolationMode.NEAREST)
        gt_edges = np.array(gt_resized) / 255.0
        
        axes[row, 0].imshow(img_resized); axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_edges, cmap="gray"); axes[row, 1].axis("off")
        axes[row, 2].imshow(pred, cmap="gray"); axes[row, 2].axis("off")
        
    plt.tight_layout()
    plt.savefig("dexined_comparison.png")
    plt.show()

# %% [markdown]
# ## 4. Deep Supervision Multi-Scale Outputs

# %%
if test_images:
    sample_tensor = TF.normalize(TF.to_tensor(TF.resize(Image.open(test_images[0]).convert("RGB"), [256, 256])), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model.train() # Temporarily enable training mode to get all 7 outputs
    with torch.no_grad():
        outputs = model(sample_tensor.unsqueeze(0).to(device))
    model.eval()
    
    fig, axes = plt.subplots(1, 7, figsize=(20, 4))
    titles = ["Side 1", "Side 2", "Side 3", "Side 4", "Side 5", "Side 6", "Fused"]
    for i, (out, title) in enumerate(zip(outputs, titles)):
        pred = torch.sigmoid(out).squeeze().cpu().numpy()
        axes[i].imshow(pred, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")
    plt.suptitle("DexiNed Multi-Scale Edges", fontweight="bold")
    plt.tight_layout()
    plt.savefig("dexined_scales.png")
    plt.show()

print("Evaluation complete!")
