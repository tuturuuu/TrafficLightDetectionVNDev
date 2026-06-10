#!/usr/bin/env python3
import argparse
import os
import cv2
import torch
import numpy as np

IMG_SIZE = 160

import torch.nn as nn


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
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def load_image(path, img_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img).unsqueeze(0)
    return tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to .jpg image to run")
    p.add_argument("--model", default="tile_proposal_cnn_model.pth", help="Path to model .pth file")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default=None, help="cpu or cuda (auto if omitted)")

    args = p.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return

    model = TileProposalCNN().to(device)
    state = torch.load(args.model, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # allow loading if file contains a dict with keys like 'model_state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            raise

    model.eval()

    inp = load_image(args.image).to(device)

    with torch.no_grad():
        logits = model(inp).squeeze(1)
        prob = torch.sigmoid(logits).cpu().item()

    kept = prob > args.threshold

    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Probability: {prob:.4f}")
    print(f"Threshold: {args.threshold}")
    print("Kept: Yes" if kept else "Kept: No")


if __name__ == "__main__":
    main()
