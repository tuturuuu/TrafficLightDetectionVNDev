import os
import cv2
import numpy as np

from glob import glob
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from joblib import dump

# -----------------------------
# CONFIG
# -----------------------------

IMAGE_DIR = "/home/vietpham/dataset/dataset/train_tiled/images"
LABEL_DIR = "/home/vietpham/dataset/dataset/train_tiled/labels"

TOP_K = 2

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = gray.shape

    features = []

    # =====================================================
    # 1. EDGE FEATURES
    # =====================================================

    edges = cv2.Canny(gray, 100, 200)

    edge_density = np.mean(edges > 0)

    features.append(edge_density)

    # =====================================================
    # 2. COLOR FEATURES
    # =====================================================
    patch_size_h = h // 4
    patch_size_w = w // 4

    red_scores = []
    green_scores = []
    yellow_scores = []
    bright_scores = []
    edge_scores = []


    # red
    red1 = cv2.inRange(
        hsv,
        (0, 100, 100),
        (10, 255, 255)
    )

    red2 = cv2.inRange(
        hsv,
        (170, 100, 100),
        (180, 255, 255)
    )

    red_mask = red1 | red2

    # green
    green_mask = cv2.inRange(
        hsv,
        (35, 80, 80),
        (90, 255, 255)
    )

    # yellow
    yellow_mask = cv2.inRange(
        hsv,
        (15, 80, 80),
        (35, 255, 255)
    )

    for py in range(4):
        for px in range(4):

            y1 = py * patch_size_h
            y2 = y1 + patch_size_h

            x1 = px * patch_size_w
            x2 = x1 + patch_size_w

            patch_red = red_mask[y1:y2, x1:x2]
            patch_green = green_mask[y1:y2, x1:x2]
            patch_yellow = yellow_mask[y1:y2, x1:x2]

            patch_gray = gray[y1:y2, x1:x2]
            patch_edge = edges[y1:y2, x1:x2]

            red_scores.append(
                np.mean(patch_red > 0)
            )

            green_scores.append(
                np.mean(patch_green > 0)
            )

            yellow_scores.append(
                np.mean(patch_yellow > 0)
            )

            bright_scores.append(
                np.mean(patch_gray > 220)
            )

            edge_scores.append(
                np.mean(patch_edge > 0)
            )

    # localized maxima
    max_red = np.max(red_scores)
    max_green = np.max(green_scores)
    max_yellow = np.max(yellow_scores)

    max_bright = np.max(bright_scores)
    max_edge = np.max(edge_scores)

    # top-k local responses
    top3_red = np.mean(
        sorted(red_scores)[-3:]
    )

    top3_green = np.mean(
        sorted(green_scores)[-3:]
    )

    features.extend([
        max_red,
        max_green,
        max_yellow,
        max_bright,
        max_edge,
        top3_red,
        top3_green
    ])

    # =====================================================
    # 3. BRIGHT PIXELS
    # =====================================================

    bright_mask = gray > 220

    bright_ratio = np.mean(bright_mask)

    features.append(bright_ratio)

    # =====================================================
    # 4. SMALL BRIGHT BLOBS
    # =====================================================

    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(
            bright_mask.astype(np.uint8)
        )

    small_blobs = 0
    medium_blobs = 0
    max_blob_area = 0

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        max_blob_area = max(
            max_blob_area,
            area
        )

        if 2 <= area <= 20:
            small_blobs += 1

        if 20 < area <= 80:
            medium_blobs += 1

    features.extend([
        small_blobs,
        medium_blobs,
        max_blob_area
    ])

    # =====================================================
    # 5. LOCALIZED PATCH FEATURES
    # =====================================================

    patch_h = h // 4
    patch_w = w // 4

    patch_brightness = []
    patch_edges = []

    for py in range(4):
        for px in range(4):

            y1 = py * patch_h
            y2 = y1 + patch_h

            x1 = px * patch_w
            x2 = x1 + patch_w

            patch_gray = gray[y1:y2, x1:x2]
            patch_edge = edges[y1:y2, x1:x2]

            patch_brightness.append(
                patch_gray.mean()
            )

            patch_edges.append(
                np.mean(patch_edge > 0)
            )

    features.extend([
        np.max(patch_brightness),
        np.std(patch_brightness),
        np.max(patch_edges)
    ])

    # =====================================================
    # 6. UPPER REGION PRIORS
    # =====================================================

    upper_half = gray[:h//2]

    upper_bright_ratio = np.mean(
        upper_half > 220
    )

    features.append(
        upper_bright_ratio
    )

    return np.array(
        features,
        dtype=np.float32
    )

# -----------------------------
# LOAD DATASET
# -----------------------------

X = []
y = []
tile_names = []

image_paths = glob(
    os.path.join(
        IMAGE_DIR,
        "*.jpg"
    )
)

for img_path in image_paths:

    filename = os.path.basename(img_path)
    stem = os.path.splitext(filename)[0]

    label_path = os.path.join(
        LABEL_DIR,
        stem + ".txt"
    )

    img = cv2.imread(img_path)

    if img is None:
        continue

    feats = extract_features(img)

    has_object = 0

    if os.path.exists(label_path):

        with open(label_path, "r") as f:

            if len(f.readlines()) > 0:
                has_object = 1

    X.append(feats)
    y.append(has_object)
    tile_names.append(stem)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("Positive ratio:", y.mean())

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

(
    X_train,
    X_test,
    y_train,
    y_test,
    names_train,
    names_test
) = train_test_split(
    X,
    y,
    tile_names,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# MODEL
# -----------------------------

clf = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "lr",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

clf.fit(X_train, y_train)

# -----------------------------
# PROBABILITIES
# -----------------------------

probs = clf.predict_proba(X_test)[:, 1]

# -----------------------------
# GROUP BY ORIGINAL IMAGE
# -----------------------------

groups = defaultdict(list)

for prob, label, name in zip(
    probs,
    y_test,
    names_test
):

    base = name.rsplit(
        "_tile",
        1
    )[0]

    groups[base].append(
        (prob, label, name)
    )

# -----------------------------
# TOP-K EVALUATION
# -----------------------------

total_tiles = 0
tiles_kept = 0

total_positive_tiles = 0
recovered_positive_tiles = 0

for base, items in groups.items():

    items = sorted(
        items,
        key=lambda x: x[0],
        reverse=True
    )

    selected = items[:TOP_K]

    selected_names = set(
        x[2] for x in selected
    )

    for prob, label, name in items:

        total_tiles += 1

        if name in selected_names:
            tiles_kept += 1

        if label == 1:

            total_positive_tiles += 1

            if name in selected_names:
                recovered_positive_tiles += 1

# -----------------------------
# RESULTS
# -----------------------------

tile_reduction = 1 - (
    tiles_kept / total_tiles
)

positive_recall = (
    recovered_positive_tiles /
    total_positive_tiles
)

print("\n===== RESULTS =====")

print(f"TOP_K: {TOP_K}")

print(f"Tile reduction: {tile_reduction:.4f}")

print(f"Positive tile recall: {positive_recall:.4f}")

print(
    f"Tiles kept: "
    f"{tiles_kept}/{total_tiles}"
)

# -----------------------------
# SAVE
# -----------------------------

dump(
    clf,
    "tile_classifier.joblib"
)

print("\nSaved model.")