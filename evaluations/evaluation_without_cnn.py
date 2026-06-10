#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/home/vietpham/dataset/dataset")
DEFAULT_YOLO_MODEL = BASE_DIR / "runs/detect/new/yolo26_traffic_light_dataset2_tiling3/weights/best.pt"

DEFAULT_TILE_SIZE = 740
DEFAULT_OVERLAP = 0.2
DEFAULT_YOLO_CONF = 0.25
DEFAULT_NMS_IOU = 0.5
EVAL_IOUV = np.linspace(0.5, 0.95, 10)


def parse_args():
    parser = argparse.ArgumentParser(description="Tiled YOLO evaluation without proposal filtering")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--yolo-model", default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--yolo-conf", type=float, default=DEFAULT_YOLO_CONF)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0, help="Limit the number of images for a quick run")
    parser.add_argument("--agnostic-nms", action="store_true")
    return parser.parse_args()


def resolve_ultralytics_device(device_arg):
    if device_arg is not None:
        if device_arg.startswith("cuda"):
            suffix = device_arg.split(":", 1)[1] if ":" in device_arg else "0"
            return int(suffix) if suffix.isdigit() else device_arg
        return device_arg
    return 0 if torch.cuda.is_available() else "cpu"


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


def yolo_predictions_on_tiles(yolo_model, tiles, yolo_conf, imgsz, device):
    if not tiles:
        return []

    tile_images = [tile["image"] for tile in tiles]
    results = yolo_model.predict(
        source=tile_images,
        conf=yolo_conf,
        imgsz=imgsz,
        verbose=False,
        device=device,
    )

    detections = []
    for tile, result in zip(tiles, results):
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            detections.append({
                "box": np.array([
                    box[0] + tile["x"],
                    box[1] + tile["y"],
                    box[2] + tile["x"],
                    box[3] + tile["y"],
                ], dtype=np.float32),
                "conf": float(conf),
                "cls": int(cls),
            })

    return detections


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
    ultra_device = resolve_ultralytics_device(args.device)

    dataset_root = Path(args.dataset_root)
    yolo_model_path = Path(args.yolo_model)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")

    images = gather_images(dataset_root)
    if args.max_images and args.max_images > 0:
        images = images[:args.max_images]

    if not images:
        raise RuntimeError(f"No images found under {dataset_root / 'train/images'}, {dataset_root / 'val/images'}, or {dataset_root / 'test/images'}")

    yolo_model = load_yolo_model(str(yolo_model_path))

    warmup_image = cv2.imread(str(images[0]))
    if warmup_image is not None:
        warmup_tiles = build_tiles(warmup_image, args.tile_size, args.overlap)
        if warmup_tiles:
            _ = yolo_predictions_on_tiles(yolo_model, warmup_tiles[:1], args.yolo_conf, args.imgsz, ultra_device)

    all_predictions = []
    all_targets = []

    total_tiles = 0
    total_detections_tiles = 0

    start_time = time.perf_counter()

    time_io = 0
    time_tiling = 0
    time_yolo = 0
    time_proposal = 0
    time_nms = 0
    time_gt = 0
    
    with torch.no_grad():
        for image_path in images:
            t0 = time.perf_counter()
            image = cv2.imread(str(image_path))
            time_io += time.perf_counter() - t0
            if image is None:
                continue

            t0 = time.perf_counter()
            tiles = build_tiles(image, args.tile_size, args.overlap)
            time_tiling += time.perf_counter() - t0
            total_tiles += len(tiles)

            t0 = time.perf_counter()
            detections = yolo_predictions_on_tiles(yolo_model, tiles, args.yolo_conf, args.imgsz, ultra_device)
            time_yolo += time.perf_counter() - t0
            total_detections_tiles += len(tiles)

            t0 = time.perf_counter()
            merged_detections = nms_detections(
                detections,
                iou_threshold=args.nms_iou,
                agnostic=args.agnostic_nms,
                max_det=300,
            )
            time_nms += time.perf_counter() - t0

            all_predictions.append(merged_detections)
            t0 = time.perf_counter()
            all_targets.append(load_ground_truth(image_path))
            time_gt += time.perf_counter() - t0

    total_time = time.perf_counter() - start_time

    evaluation = evaluate_predictions(all_predictions, all_targets, EVAL_IOUV)

    image_count = len(all_predictions)
    tile_fps = total_detections_tiles / total_time if total_time > 0 else 0.0
    real_fps = image_count / total_time if total_time > 0 else 0.0

    print(f"\n===== TIME BREAKDOWN =====")
    print(f"I/O (imread):        {time_io:7.2f}s ({time_io/total_time*100:5.1f}%)")
    print(f"Tiling:              {time_tiling:7.2f}s ({time_tiling/total_time*100:5.1f}%)")
    print(f"Proposal CNN:        {time_proposal:7.2f}s ({time_proposal/total_time*100:5.1f}%)")
    print(f"YOLO inference:      {time_yolo:7.2f}s ({time_yolo/total_time*100:5.1f}%)")
    print(f"NMS:                 {time_nms:7.2f}s ({time_nms/total_time*100:5.1f}%)")
    print(f"Ground truth load:   {time_gt:7.2f}s ({time_gt/total_time*100:5.1f}%)")
    print(f"Other/overhead:      {total_time - (time_io+time_tiling+time_proposal+time_yolo+time_nms+time_gt):7.2f}s")
    print()
    print(f"Images processed: {image_count}")
    print(f"Tiles generated: {total_tiles}")
    print(f"Tiles kept: {total_tiles}")
    print(f"Tile reduction: 0.0000")
    print(f"Total time: {total_time:.3f} sec")
    print(f"Tile-level FPS: {tile_fps:.2f}")
    print(f"REAL image-level FPS (with tiling): {real_fps:.2f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"mAP50: {evaluation['map50']:.4f}")
    print(f"mAP50-95: {evaluation['map50_95']:.4f}")
    


if __name__ == "__main__":
    main()
