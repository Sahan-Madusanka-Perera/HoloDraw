# %% [markdown]
# # HoloPaint DexiNed Edge Detection Model — Training Pipeline
#
# This notebook trains a **DexiNed (Dense Extreme Inception Network)** for high-fidelity
# edge/contour detection on the **BSDS500** dataset. The trained model is exported as
# `holopaint_dexined.pth` for use in the HoloPaint application.
#
# ## Architecture Overview
# - **Backbone**: Custom convolutional encoder
# - **Deep Supervision**: 6 side outputs from intermediate layers
# - **Fusion Layer**: Combines all 6 side outputs into a final crisp edge map
#
# ## Key Differences from U-Net
# 1. Natively outputs incredibly thin, clean lines ("coloring book" style).
# 2. Uses Deep Supervision (loss is calculated at 7 different points in the network).
# 3. No need for complex morphological gap-closing in post-processing.

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image
from sklearn.metrics import f1_score

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Dataset Setup (BIPED)

# %%
DATASET_DIR = Path("./data")
BIPED_DIR = DATASET_DIR / "BIPED" / "edges"

if not BIPED_DIR.exists():
    print("WARNING: BIPED dataset not found at expected location.")
else:
    print(f"BIPED dataset found at {BIPED_DIR}")

class BIPEDDataset(Dataset):
    def __init__(self, split="train", img_size=256, augment=True):
        self.img_size = img_size
        self.augment = augment
        
        # BIPED does not have a formal 'val' split in this standard format, 
        # so we will use 'test' for validation if 'val' is requested.
        dir_split = "test" if split == "val" else split
        
        if dir_split == "train":
            img_dir = BIPED_DIR / "imgs" / dir_split / "rgbr" / "real"
            gt_dir = BIPED_DIR / "edge_maps" / dir_split / "rgbr" / "real"
        else:
            img_dir = BIPED_DIR / "imgs" / dir_split / "rgbr"
            gt_dir = BIPED_DIR / "edge_maps" / dir_split / "rgbr"
        
        self.image_paths = sorted(img_dir.glob("*.jpg"))
        self.gt_paths = sorted(gt_dir.glob("*.png"))
        
        # Assert matching
        if len(self.image_paths) != len(self.gt_paths):
            print(f"Warning: mismatch between images ({len(self.image_paths)}) and edges ({len(self.gt_paths)}) for {split}")
            
        print(f"[{split}] Loaded {len(self.image_paths)} image-GT pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        edge_pil = Image.open(self.gt_paths[idx]).convert("L")

        # Transforms
        resize_to = self.img_size + 32 if self.augment else self.img_size
        image = TF.resize(image, [resize_to, resize_to])
        edge_pil = TF.resize(edge_pil, [resize_to, resize_to], interpolation=T.InterpolationMode.NEAREST)

        if self.augment:
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image = TF.crop(image, i, j, h, w)
            edge_pil = TF.crop(edge_pil, i, j, h, w)
            if random.random() > 0.5:
                image, edge_pil = TF.hflip(image), TF.hflip(edge_pil)
            if random.random() > 0.5:
                image, edge_pil = TF.vflip(image), TF.vflip(edge_pil)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image, edge_pil = TF.rotate(image, angle), TF.rotate(edge_pil, angle)
            
            # Advanced Augmentation: Color Jitter
            if random.random() > 0.5:
                jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = jitter(image)
            
            # Advanced Augmentation: Blur
            if random.random() > 0.7:
                image = TF.gaussian_blur(image, kernel_size=[3, 3])
        else:
            image = TF.center_crop(image, [self.img_size, self.img_size])
            edge_pil = TF.center_crop(edge_pil, [self.img_size, self.img_size])

        image = TF.normalize(TF.to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Binarize edge
        edge = TF.to_tensor(edge_pil)
        edge = (edge > 0.5).float()
        
        return image, edge

train_loader = DataLoader(BIPEDDataset("train", augment=True), batch_size=4, shuffle=True, drop_last=True)
val_loader = DataLoader(BIPEDDataset("val", augment=False), batch_size=2, shuffle=False)
test_loader = DataLoader(BIPEDDataset("test", augment=False), batch_size=2, shuffle=False)

# %% [markdown]
# ## 3. DexiNed Model Architecture

# %%
class DoubleConv(nn.Module):
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

class DexiNed(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.enc5 = DoubleConv(256, 512)
        self.enc6 = DoubleConv(512, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.side1 = nn.Conv2d(32, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.side3 = nn.Conv2d(128, 1, kernel_size=1)
        self.side4 = nn.Conv2d(256, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)
        self.side6 = nn.Conv2d(512, 1, kernel_size=1)

        self.fuse = nn.Conv2d(6, 1, kernel_size=1)

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
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 4. Deep Supervision Loss Function

# %%
pos_weight_tensor = torch.tensor([10.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

def dexined_loss(preds, target):
    """Calculates weighted BCE loss across all 7 deep supervision outputs."""
    loss = 0.0
    for pred in preds:
        loss += criterion(pred, target)
    return loss

# %% [markdown]
# ## 5. Training Loop

# %%
NUM_EPOCHS = 30
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

train_losses, val_losses, val_f1_scores = [], [], []
best_val_loss = float('inf')

print("Starting DexiNed Training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for images, edges in train_loader:
        images, edges = images.to(device), edges.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)  # Returns list of 7 logits
        loss = dexined_loss(outputs, edges)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, edges in val_loader:
            images, edges = images.to(device), edges.to(device)
            # To compute validation loss, we temporarily set training=True to get all outputs
            model.train()
            outputs = model(images)
            model.eval()
            
            loss = dexined_loss(outputs, edges)
            val_loss += loss.item()

            # For F1 score, we only care about the final fused output (sigmoid applied)
            fused_pred = model(images)
            preds = (fused_pred > 0.5).float()
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(edges.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    f1 = f1_score(np.concatenate(all_targets), np.concatenate(all_preds), zero_division=0)
    val_f1_scores.append(f1)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_dexined.pth")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {f1:.4f}")

print("Training complete!")

# %% [markdown]
# ## 6. Export Model

# %%
model.load_state_dict(torch.load("best_dexined.pth", map_location=device))
export_path = "holopaint_dexined.pth"
torch.save(model.state_dict(), export_path)
print(f"Model exported to {export_path}")

torch.save({
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_f1_scores": val_f1_scores
}, "dexined_training_log.pth")
print("Logs saved to dexined_training_log.pth")
