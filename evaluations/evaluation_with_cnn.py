#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/home/vietpham/dataset/dataset")
DEFAULT_TILE_MODEL = BASE_DIR / "../cnn_classifier/tile_proposal_cnn_model.pth"
DEFAULT_YOLO_MODEL = BASE_DIR / "../runs/detect/new/yolo26_traffic_light_dataset2_tiling3/weights/best.pt"

IMG_SIZE = 160
DEFAULT_TILE_SIZE = 740
DEFAULT_OVERLAP = 0.2
DEFAULT_TILE_THRESHOLD = 0.5
DEFAULT_YOLO_CONF = 0.25
DEFAULT_NMS_IOU = 0.5
DEFAULT_PROPOSAL_BATCH_SIZE = 64
DEFAULT_YOLO_BATCH_SIZE = 32

EVAL_IOUV = np.linspace(0.5, 0.95, 10)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Tile proposal + YOLO evaluation on original dataset images")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--tile-model", default=str(DEFAULT_TILE_MODEL))
    parser.add_argument("--yolo-model", default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--tile-threshold", type=float, default=DEFAULT_TILE_THRESHOLD)
    parser.add_argument("--yolo-conf", type=float, default=DEFAULT_YOLO_CONF)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0, help="Limit the number of images for a quick run")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--proposal-batch-size", type=int, default=DEFAULT_PROPOSAL_BATCH_SIZE,
                        help="Max tiles per proposal-CNN batch. Batches never span more than one "
                             "image's tiles; this only caps how many of that one image's tiles "
                             "are scored in a single forward pass (raise it if you want every "
                             "image's tiles scored in one shot regardless of tile count).")
    parser.add_argument("--yolo-batch-size", type=int, default=DEFAULT_YOLO_BATCH_SIZE,
                        help="Max tiles per YOLO batch. Batches never span more than one image's "
                             "kept tiles; this only caps how many of that one image's tiles are "
                             "run through YOLO in a single forward pass.")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg is not None:
        if device_arg.startswith("cuda"):
            return torch.device(device_arg)
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_ultralytics_device(device_arg):
    if device_arg is not None:
        if device_arg.startswith("cuda"):
            suffix = device_arg.split(":", 1)[1] if ":" in device_arg else "0"
            return int(suffix) if suffix.isdigit() else device_arg
        return device_arg
    return 0 if torch.cuda.is_available() else "cpu"


def load_state_dict(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if isinstance(checkpoint, dict):
        cleaned = {}
        for key, value in checkpoint.items():
            cleaned[key.replace("module.", "", 1)] = value
        checkpoint = cleaned

    model.load_state_dict(checkpoint, strict=True)


def load_tile_model(model_path, device):
    model = TileProposalCNN().to(device)
    load_state_dict(model, model_path, device)
    model.eval()
    return model


def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO inference. Install it in the active environment before running this script."
        ) from exc

    return YOLO(model_path)


def gather_images(dataset_root):
    dataset_root = Path(dataset_root)
    image_paths = []

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split / "images"
        if not split_dir.exists():
            continue
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(sorted(split_dir.glob(pattern)))

    return image_paths


def get_tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    step = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, step))
    last_valid = length - tile_size

    if not starts or starts[-1] < last_valid:
        starts.append(last_valid)

    return starts


def pad_to_tile(tile, tile_size):
    if tile.shape[0] >= tile_size and tile.shape[1] >= tile_size:
        return tile

    bottom = tile_size - tile.shape[0]
    right = tile_size - tile.shape[1]
    return cv2.copyMakeBorder(tile, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def build_tiles(image, tile_size, overlap):
    h, w = image.shape[:2]
    tiles = []

    for top in get_tile_starts(h, tile_size, overlap):
        for left in get_tile_starts(w, tile_size, overlap):
            tile = image[top:top + tile_size, left:left + tile_size]
            tile = pad_to_tile(tile, tile_size)
            tiles.append({"image": tile, "x": left, "y": top})

    return tiles


def preprocess_tiles_gpu(tile_images, device, target_size=IMG_SIZE):
    """
    Preprocess tiles on GPU for proposal CNN.
    
    Args:
        tile_images: List of numpy arrays (H, W, 3) in BGR format
        device: torch device
        target_size: target size for resizing
    
    Returns:
        Preprocessed batch tensor (N, 3, target_size, target_size)
    """
    if not tile_images:
        return torch.empty((0, 3, target_size, target_size), device=device)
    
    # Stack numpy arrays
    batch = np.stack(tile_images, axis=0)  # (N, H, W, 3)
    
    # Convert to torch tensor and move to GPU
    batch = torch.from_numpy(batch).to(device)  # (N, H, W, 3)
    
    # Permute to (N, 3, H, W) for PyTorch
    batch = batch.permute(0, 3, 1, 2).contiguous()
    
    # Convert BGR to RGB by flipping channel dimension
    batch = batch.flip(1)
    
    # Resize on GPU
    if batch.shape[2] != target_size or batch.shape[3] != target_size:
        batch = F.interpolate(
            batch.float(),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
    else:
        batch = batch.float()
    
    # Normalize to [0, 1]
    batch = batch / 255.0
    
    return batch


def tile_model_scores_batched(tile_model, tile_images, device, batch_size=6):
    """
    Score tiles using the proposal CNN with efficient batching.
    
    Args:
        tile_model: Proposal CNN model
        tile_images: List of tile images (numpy arrays)
        device: torch device
        batch_size: batch size for inference
    
    Returns:
        numpy array of probabilities
    """
    if not tile_images:
        return np.zeros((0,), dtype=np.float32)
    
    all_probs = []
    
    for i in range(0, len(tile_images), batch_size):
        batch_tiles = tile_images[i:i + batch_size]
        
        # Preprocess on GPU
        batch = preprocess_tiles_gpu(batch_tiles, device, IMG_SIZE)
        
        # Inference
        with torch.no_grad():
            logits = tile_model(batch).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        all_probs.append(probs)
    
    return np.concatenate(all_probs, axis=0)


def yolo_predictions_batched(
    yolo_model,
    tile_images,
    tile_metadata,
    yolo_conf,
    imgsz,
    device,
    batch_size=6,
):
    """
    Returns:
        List[List[detection]]
        One list per tile
    """
    if not tile_images:
        return []

    per_tile_detections = []

    for i in range(0, len(tile_images), batch_size):
        batch_tiles = tile_images[i:i + batch_size]
        batch_metadata = tile_metadata[i:i + batch_size]

        results = yolo_model.predict(
            source=batch_tiles,
            conf=yolo_conf,
            imgsz=imgsz,
            verbose=False,
            device=device,
        )

        for tile_meta, result in zip(batch_metadata, results):

            tile_dets = []

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confs, classes):

                    tile_dets.append({
                        "box": np.array([
                            box[0] + tile_meta["x"],
                            box[1] + tile_meta["y"],
                            box[2] + tile_meta["x"],
                            box[3] + tile_meta["y"],
                        ], dtype=np.float32),
                        "conf": float(conf),
                        "cls": int(cls),
                    })

            per_tile_detections.append(tile_dets)

    return per_tile_detections


def box_iou(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    box_area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - inter
    return inter / np.maximum(union, 1e-9)


def nms_single_class(boxes, scores, iou_threshold, max_det):
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(current)

        if max_det and len(keep) >= max_det:
            break

        if order.size == 1:
            break

        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)


def nms_detections(detections, iou_threshold=0.5, agnostic=False, max_det=300):
    if not detections:
        return []

    boxes = np.stack([det["box"] for det in detections], axis=0)
    scores = np.array([det["conf"] for det in detections], dtype=np.float32)
    classes = np.array([det["cls"] for det in detections], dtype=np.int32)

    kept_indices = []

    if agnostic:
        kept_indices = nms_single_class(boxes, scores, iou_threshold, max_det)
    else:
        for cls in np.unique(classes):
            cls_indices = np.where(classes == cls)[0]
            if cls_indices.size == 0:
                continue

            cls_keep = nms_single_class(boxes[cls_indices], scores[cls_indices], iou_threshold, max_det)
            kept_indices.extend(cls_indices[cls_keep].tolist())

        kept_indices = np.array(sorted(kept_indices, key=lambda idx: scores[idx], reverse=True), dtype=np.int64)

        if max_det and kept_indices.size > max_det:
            kept_indices = kept_indices[:max_det]

    return [detections[idx] for idx in kept_indices]


def load_ground_truth(image_path):
    image_path = Path(image_path)
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"

    if not label_path.exists():
        return []

    image = cv2.imread(str(image_path))
    if image is None:
        return []

    h, w = image.shape[:2]
    targets = []

    with open(label_path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            x2 = (xc + bw / 2.0) * w
            y2 = (yc + bh / 2.0) * h

            targets.append({
                "cls": int(cls),
                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            })

    return targets


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])


def evaluate_predictions(all_predictions, all_targets, iouv):
    class_ids = sorted(
        set(
            [pred["cls"] for image_preds in all_predictions for pred in image_preds]
            + [target["cls"] for image_targets in all_targets for target in image_targets]
        )
    )

    ap = np.zeros((len(class_ids), len(iouv)), dtype=np.float32)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for class_index, class_id in enumerate(class_ids):
        class_predictions = []
        class_targets_by_image = []
        num_targets = 0

        for image_index, (image_predictions, image_targets) in enumerate(zip(all_predictions, all_targets)):
            gt_boxes = [target["box"] for target in image_targets if target["cls"] == class_id]
            class_targets_by_image.append(gt_boxes)
            num_targets += len(gt_boxes)

            for prediction in image_predictions:
                if prediction["cls"] == class_id:
                    class_predictions.append((image_index, prediction["conf"], prediction["box"]))

        if num_targets == 0:
            continue

        class_predictions.sort(key=lambda item: item[1], reverse=True)

        for iou_index, iou_threshold in enumerate(iouv):
            matched = [np.zeros(len(gt_boxes), dtype=bool) for gt_boxes in class_targets_by_image]
            tp = np.zeros(len(class_predictions), dtype=np.float32)
            fp = np.zeros(len(class_predictions), dtype=np.float32)

            for prediction_index, (image_index, _, pred_box) in enumerate(class_predictions):
                gt_boxes = class_targets_by_image[image_index]
                if not gt_boxes:
                    fp[prediction_index] = 1.0
                    continue

                gt_boxes_array = np.stack(gt_boxes, axis=0)
                ious = box_iou(pred_box, gt_boxes_array)
                best_index = int(np.argmax(ious))

                if ious[best_index] >= iou_threshold and not matched[image_index][best_index]:
                    tp[prediction_index] = 1.0
                    matched[image_index][best_index] = True
                else:
                    fp[prediction_index] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / (num_targets + 1e-9)
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap[class_index, iou_index] = compute_ap(recall, precision)

            if iou_index == 0:
                total_tp += int(tp.sum())
                total_fp += int(fp.sum())
                total_fn += num_targets - int(tp.sum())

    valid_classes = [i for i, class_id in enumerate(class_ids) if any(target["cls"] == class_id for image_targets in all_targets for target in image_targets)]
    if valid_classes:
        map50 = float(np.mean(ap[valid_classes, 0]))
        map5095 = float(np.mean(ap[valid_classes].mean(axis=1)))
    else:
        map50 = 0.0
        map5095 = 0.0

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "map50": map50,
        "map50_95": map5095,
        "classes_evaluated": class_ids,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    ultra_device = resolve_ultralytics_device(args.device)

    dataset_root = Path(args.dataset_root)
    tile_model_path = Path(args.tile_model)
    yolo_model_path = Path(args.yolo_model)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not tile_model_path.exists():
        raise FileNotFoundError(f"Tile model not found: {tile_model_path}")
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")

    images = gather_images(dataset_root)
    if args.max_images and args.max_images > 0:
        images = images[:args.max_images]

    if not images:
        raise RuntimeError(f"No images found under {dataset_root / 'train/images'}, {dataset_root / 'val/images'}, or {dataset_root / 'test/images'}")

    print(f"Loading models...")
    tile_model = load_tile_model(tile_model_path, device)
    yolo_model = load_yolo_model(str(yolo_model_path))

    # Warmup
    print(f"Warming up models...")
    warmup_image = cv2.imread(str(images[0]))
    if warmup_image is not None:
        warmup_tiles = build_tiles(warmup_image, args.tile_size, args.overlap)
        if warmup_tiles:
            warmup_images = [t["image"] for t in warmup_tiles[:min(4, len(warmup_tiles))]]
            _ = tile_model_scores_batched(tile_model, warmup_images, device, args.proposal_batch_size)
            _ = yolo_predictions_batched(yolo_model, warmup_images[:1], warmup_tiles[:1], 
                                        args.yolo_conf, args.imgsz, ultra_device, args.yolo_batch_size)

    print(f"\nProcessing {len(images)} images...")
    print(f"Proposal batch cap (per image): {args.proposal_batch_size}")
    print(f"YOLO batch cap (per image): {args.yolo_batch_size}")
    print(f"Tile threshold: {args.tile_threshold}\n")

    # Timing counters
    time_io = 0.0
    time_tiling = 0.0
    time_proposal = 0.0
    time_yolo = 0.0
    time_nms = 0.0
    time_gt = 0.0

    # Statistics
    total_tiles = 0
    total_kept_tiles = 0
    total_images_with_kept_tiles = 0

    # Results
    all_predictions = []
    all_targets = []

    start_time = time.perf_counter()

    # Phase 1: Load all images and build all tiles
    print("Phase 1: Loading images and building tiles...")
    all_image_data = []
    
    t0 = time.perf_counter()
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            all_image_data.append(None)
            continue
        
        all_image_data.append({
            "path": image_path,
            "image": image,
        })
    time_io += time.perf_counter() - t0

    t0 = time.perf_counter()
    for img_data in all_image_data:
        if img_data is None:
            continue
        tiles = build_tiles(img_data["image"], args.tile_size, args.overlap)
        img_data["tiles"] = tiles
        total_tiles += len(tiles)
    time_tiling += time.perf_counter() - t0

    # Phase 2: Score tiles with proposal CNN, ONE IMAGE AT A TIME.
    # Each image's own tiles (X of them) are run together as a single batch,
    # mirroring real-world deployment where one image arrives, gets cut into
    # X tiles, and those X tiles are the only thing the model sees in that
    # forward pass. Tiles from different images are never combined into the
    # same batch. If a single image produces more tiles than
    # --proposal-batch-size, it is split into multiple batches, but those
    # batches still only ever contain tiles from that one image.
    print(f"Phase 2: Scoring {total_tiles} tiles with proposal CNN (one image's tiles per batch)...")

    t0 = time.perf_counter()
    for img_data in all_image_data:
        if img_data is None:
            continue

        tile_images = [tile["image"] for tile in img_data["tiles"]]

        # batch_size = number of tiles this image produced (capped only to
        # avoid OOM on pathological images with an unusually large tile count)
        per_image_batch_size = min(len(tile_images), args.proposal_batch_size) if tile_images else 1

        img_scores = tile_model_scores_batched(tile_model, tile_images, device, per_image_batch_size)

        kept_tiles = [tile for tile, score in zip(img_data["tiles"], img_scores)
                      if score > args.tile_threshold]
        img_data["kept_tiles"] = kept_tiles
        total_kept_tiles += len(kept_tiles)
        if kept_tiles:
            total_images_with_kept_tiles += 1
    time_proposal += time.perf_counter() - t0

    # Phase 3: Run YOLO on kept tiles, ONE IMAGE AT A TIME.
    # Same principle as Phase 2: only the kept tiles that came from a single
    # image are ever batched together for YOLO inference. This means the
    # detector never sees a mix of tiles from two different images in one
    # forward pass, matching real-world streaming inference.
    print(f"Phase 3: Running YOLO on {total_kept_tiles} kept tiles (one image's tiles per batch)...")

    detections_by_image = {}

    t0 = time.perf_counter()
    for img_idx, img_data in enumerate(all_image_data):
        if img_data is None:
            detections_by_image[img_idx] = []
            continue

        kept_tiles = img_data["kept_tiles"]
        if not kept_tiles:
            detections_by_image[img_idx] = []
            continue

        tile_images = [tile["image"] for tile in kept_tiles]
        tile_metadata = [{"x": tile["x"], "y": tile["y"]} for tile in kept_tiles]

        # batch_size = number of kept tiles this image produced (capped only
        # to avoid OOM on pathological images with an unusually large count)
        per_image_batch_size = min(len(tile_images), args.yolo_batch_size)

        per_tile_detections = yolo_predictions_batched(
            yolo_model,
            tile_images,
            tile_metadata,
            args.yolo_conf,
            args.imgsz,
            ultra_device,
            per_image_batch_size,
        )

        image_detections = []
        for tile_dets in per_tile_detections:
            image_detections.extend(tile_dets)
        detections_by_image[img_idx] = image_detections
    time_yolo += time.perf_counter() - t0

    # Phase 4: NMS and ground truth loading per image
    print("Phase 4: NMS and evaluation...")
    
    for img_idx, img_data in enumerate(all_image_data):
        if img_data is None:
            all_predictions.append([])
            all_targets.append([])
            continue
        
        detections = detections_by_image[img_idx]
        
        # NMS
        t0 = time.perf_counter()
        merged_detections = nms_detections(
            detections,
            iou_threshold=args.nms_iou,
            agnostic=args.agnostic_nms,
            max_det=300,
        )
        time_nms += time.perf_counter() - t0
        
        # Ground truth
        t0 = time.perf_counter()
        all_predictions.append(merged_detections)
        all_targets.append(load_ground_truth(img_data["path"]))
        time_gt += time.perf_counter() - t0

    total_time = time.perf_counter() - start_time

    # Evaluation
    print("\nEvaluating predictions...")
    evaluation = evaluate_predictions(all_predictions, all_targets, EVAL_IOUV)

    image_count = len(all_predictions)
    tile_fps = total_kept_tiles / total_time if total_time > 0 else 0.0
    real_fps = image_count / (total_time - time_gt - time_io) if total_time > 0 else 0.0
    tile_reduction = 1.0 - (total_kept_tiles / total_tiles) if total_tiles > 0 else 0.0

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Images processed: {image_count}")
    print(f"Tiles generated: {total_tiles}")
    print(f"Tiles kept by proposal model: {total_kept_tiles}")
    print(f"Images with at least one kept tile: {total_images_with_kept_tiles}")
    print(f"Tile reduction: {tile_reduction:.4f}")
    print(f"Total time: {total_time:.3f} sec")
    print(f"Tile-level FPS: {tile_fps:.2f}")
    print(f"REAL image-level FPS (with tiling): {real_fps:.2f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"mAP50: {evaluation['map50']:.4f}")
    print(f"mAP50-95: {evaluation['map50_95']:.4f}")

    # Time breakdown
    print(f"\n===== TIME BREAKDOWN =====")
    print(f"I/O (imread):        {time_io:7.2f}s ({time_io/total_time*100:5.1f}%)")
    print(f"Tiling:              {time_tiling:7.2f}s ({time_tiling/total_time*100:5.1f}%)")
    print(f"Proposal CNN:        {time_proposal:7.2f}s ({time_proposal/total_time*100:5.1f}%)")
    print(f"YOLO inference:      {time_yolo:7.2f}s ({time_yolo/total_time*100:5.1f}%)")
    print(f"NMS:                 {time_nms:7.2f}s ({time_nms/total_time*100:5.1f}%)")
    print(f"Ground truth load:   {time_gt:7.2f}s ({time_gt/total_time*100:5.1f}%)")
    other_time = total_time - (time_io + time_tiling + time_proposal + time_yolo + time_nms + time_gt)
    print(f"Other/overhead:      {other_time:7.2f}s ({other_time/total_time*100:5.1f}%)")


if __name__ == "__main__":
    main()