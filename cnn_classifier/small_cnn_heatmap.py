"""
Grad-CAM visualization for TileProposalCNN.

Shows which pixels in a tile the model actually used to make its
positive/negative prediction, so you can check whether it's keying off
the traffic light itself vs. tile position/composition shortcuts.

Usage:
    python gradcam_visualize.py

Outputs saved to ./gradcam_out/ :
    <stem>_orig.jpg   - original tile (resized to IMG_SIZE for reference)
    <stem>_cam.jpg     - Grad-CAM heatmap overlay
"""

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from glob import glob

# =====================================================
# CONFIG — match your training script
# =====================================================

IMG_SIZE = 64  # change if you bumped this during retraining
MODEL_PATH = "tile_proposal_cnn_model.pth"

TEST_BASE_DIR = "/home/vietpham/dataset/dataset/test_tiled"
TEST_IMAGE_DIR = os.path.join(TEST_BASE_DIR, "images")
TEST_LABEL_DIR = os.path.join(TEST_BASE_DIR, "labels")

OUT_DIR = "./gradcam_out"
NUM_SAMPLES = 12          # how many positive tiles to visualize
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# MODEL — identical definition to training script
# =====================================================

class TileProposalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# =====================================================
# GRAD-CAM
# =====================================================

class GradCAM:
    """
    Hooks the last conv block's output (before the final pool) to get
    both the activations and their gradients w.r.t. the predicted logit.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()
        # Returning a clone breaks the in-place ReLU's link to the tensor
        # we're hooking, which avoids the "view + inplace" autograd error.
        return out.clone()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad()

        logit = self.model(input_tensor)  # shape (1, 1)
        logit.backward()

        # global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, torch.sigmoid(logit).item()


# =====================================================
# LOAD MODEL
# =====================================================

model = TileProposalCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# last conv layer = last module in self.features (index -1 is ReLU,
# but hooking the ReLU output is fine — it's what feeds into the pool)
target_layer = model.features[-1]

# register_full_backward_hook conflicts with in-place ops on the SAME
# module (raises "view + inplace" RuntimeError). inplace=True is just a
# memory optimization, not a learned parameter, so disabling it here is
# safe and does not change the model's predictions.
if isinstance(target_layer, nn.ReLU) and target_layer.inplace:
    target_layer.inplace = False

cam_extractor = GradCAM(model, target_layer)

# =====================================================
# COLLECT SOME POSITIVE TEST TILES
# =====================================================

test_images = glob(os.path.join(TEST_IMAGE_DIR, "*.jpg"))

positive_samples = []

for path in test_images:
    stem = os.path.splitext(os.path.basename(path))[0]
    label_path = os.path.join(TEST_LABEL_DIR, stem + ".txt")

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            if len(f.readlines()) > 0:
                positive_samples.append(path)

    if len(positive_samples) >= NUM_SAMPLES:
        break

print(f"Found {len(positive_samples)} positive tiles to visualize "
      f"(out of {len(test_images)} total test images)")

if len(positive_samples) == 0:
    raise RuntimeError(
        "No positive tiles found — check TEST_LABEL_DIR path / label files."
    )

# =====================================================
# RUN GRAD-CAM ON EACH SAMPLE
# =====================================================

for path in positive_samples:

    stem = os.path.splitext(os.path.basename(path))[0]

    raw = cv2.imread(path)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    input_tensor = torch.tensor(img).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(False)

    cam, prob = cam_extractor.generate(input_tensor)

    # upsample CAM from conv feature map size (IMG_SIZE/4) back to IMG_SIZE
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(resized, 0.5, heatmap, 0.5, 0)

    # also make a larger version for easier viewing (upscale 6x, nearest neighbor
    # so you can still see the true pixel blockiness of the 64px input)
    scale = 6
    orig_big = cv2.resize(resized, (IMG_SIZE * scale, IMG_SIZE * scale),
                           interpolation=cv2.INTER_NEAREST)
    overlay_big = cv2.resize(overlay, (IMG_SIZE * scale, IMG_SIZE * scale),
                              interpolation=cv2.INTER_NEAREST)

    side_by_side = np.hstack([orig_big, overlay_big])
    side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)

    out_path = os.path.join(OUT_DIR, f"{stem}_prob{prob:.2f}.jpg")
    cv2.imwrite(out_path, side_by_side_bgr)

    print(f"{stem}: prob={prob:.4f} -> saved {out_path}")

print(f"\nDone. Check {OUT_DIR}/ — left half = original tile, "
      f"right half = Grad-CAM overlay (red/yellow = high influence on the "
      f"positive prediction).")