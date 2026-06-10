import os
import cv2
import torch
import random
import numpy as np

from glob import glob
from collections import defaultdict

from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# CONFIG
# =====================================================

IMAGE_DIR = "/home/vietpham/dataset/dataset/train_tiled/images"
LABEL_DIR = "/home/vietpham/dataset/dataset/train_tiled/labels"

IMG_SIZE = 64

BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3

TOP_K = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# DATASET
# =====================================================

class TileDataset(Dataset):

    def __init__(
        self,
        image_paths,
        augment=False
    ):

        self.image_paths = image_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def augment_image(self, img):

        # brightness
        if random.random() < 0.5:

            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)

            img = cv2.convertScaleAbs(
                img,
                alpha=alpha,
                beta=beta
            )

        # blur
        if random.random() < 0.3:

            img = cv2.GaussianBlur(
                img,
                (3, 3),
                0
            )

        return img

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]

        label_path = os.path.join(
            LABEL_DIR,
            stem + ".txt"
        )

        img = cv2.imread(img_path)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        if self.augment:
            img = self.augment_image(img)

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

        img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))

        label = 0.0

        if os.path.exists(label_path):

            with open(label_path, "r") as f:

                if len(f.readlines()) > 0:
                    label = 1.0

        return (
            torch.tensor(img),
            torch.tensor(label),
            stem
        )

# =====================================================
# MODEL
# =====================================================

class TileProposalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # --------------------------------
            # stage 1
            # --------------------------------

            nn.Conv2d(
                3, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # --------------------------------
            # stage 2
            # --------------------------------

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # --------------------------------
            # stage 3
            # --------------------------------

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1
            ),
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
# LOAD DATA
# =====================================================

all_images = glob(
    os.path.join(
        IMAGE_DIR,
        "*.jpg"
    )
)

groups = defaultdict(list)

for path in all_images:

    stem = os.path.splitext(
        os.path.basename(path)
    )[0]

    base = stem.rsplit("_tile", 1)[0]

    groups[base].append(path)

base_images = list(groups.keys())

train_bases, test_bases = train_test_split(
    base_images,
    test_size=0.2,
    random_state=42
)

train_imgs = []
test_imgs = []

for b in train_bases:
    train_imgs.extend(groups[b])

for b in test_bases:
    test_imgs.extend(groups[b])

train_ds = TileDataset(
    train_imgs,
    augment=True
)

test_ds = TileDataset(
    test_imgs,
    augment=False
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# =====================================================
# MODEL SETUP
# =====================================================

model = TileProposalCNN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0]).to(DEVICE)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
)

# =====================================================
# TRAIN
# =====================================================

best_loss = float('inf')
patience_counter = 0
early_stopping_patience = 10

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for imgs, labels, _ in train_loader:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(imgs).squeeze(1)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    scheduler.step(avg_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.4f}"
    )

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Save the model
torch.save(model.state_dict(), "tile_proposal_cnn_model.pth")
print("Model saved to tile_proposal_cnn_model.pth")

# =====================================================
# EVALUATE
# =====================================================

model.eval()

groups = defaultdict(list)

with torch.no_grad():

    for imgs, labels, stems in test_loader:

        imgs = imgs.to(DEVICE)

        logits = model(imgs).squeeze(1)

        probs = torch.sigmoid(logits)

        probs = probs.cpu().numpy()
        labels = labels.numpy()

        for prob, label, stem in zip(
            probs,
            labels,
            stems
        ):

            base = stem.rsplit(
                "_tile",
                1
            )[0]

            groups[base].append(
                (
                    float(prob),
                    int(label),
                    stem
                )
            )

# =====================================================
# TOP-K EVALUATION
# =====================================================

for THRESHOLD in [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
]:

    # reset counters for this threshold
    total_tiles = 0
    tiles_kept = 0

    total_positive_tiles = 0
    recovered_positive_tiles = 0

    model.eval()

    with torch.no_grad():

        for imgs, labels, stems in test_loader:

            imgs = imgs.to(DEVICE)

            logits = model(imgs).squeeze(1)

            probs = torch.sigmoid(logits)

            probs = probs.cpu().numpy()
            labels = labels.numpy()

            preds = probs > THRESHOLD

            for pred, label in zip(
                preds,
                labels
            ):

                total_tiles += 1

                if pred:
                    tiles_kept += 1

                if label == 1:

                    total_positive_tiles += 1

                    if pred:
                        recovered_positive_tiles += 1

    tile_reduction = 1 - (
        tiles_kept / total_tiles
    ) if total_tiles > 0 else 0.0

    positive_recall = (
        recovered_positive_tiles /
        total_positive_tiles
    ) if total_positive_tiles > 0 else 0.0

    print("\n===== THRESHOLD RESULTS =====")

    print(f"Threshold: {THRESHOLD}")

    print(
        f"Tile reduction: "
        f"{tile_reduction:.4f}"
    )

    print(
        f"Positive recall: "
        f"{positive_recall:.4f}"
    )

    print(
        f"Tiles kept: "
        f"{tiles_kept}/{total_tiles}"
    )