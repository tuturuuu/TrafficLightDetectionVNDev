import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = YOLO("runs/dataset2_tiling_cbam3/weights/best.pt")

# --- Reorder class names ---
original_names = model.model.names
kept_ids = list(range(16, 57))
kept_names = [original_names[i] for i in kept_ids]
num_classes = len(original_names)

new_names = {}
for new_id, name in enumerate(kept_names):
    new_names[new_id] = name
for i in range(len(kept_names), num_classes):
    new_names[i] = ''

model.model.names = new_names

# --- Dataset paths ---
dataset_path = "/home/vietpham/dataset/dataset"
val_images = sorted(glob.glob(os.path.join(dataset_path, "val/images/*.jpg")))

# --- IoU ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# --- Read GT ---
def read_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw/2) * img_w)
                y1 = int((y - bh/2) * img_h)
                x2 = int((x + bw/2) * img_w)
                y2 = int((y + bh/2) * img_h)
                boxes.append([cls, x1, y1, x2, y2])
    return boxes

# --- Draw GT (BIGGER STYLE) ---
def draw_yolo_boxes(image_path, label_path, color=(0, 255, 0), label_type="GT"):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    font_scale = 1.2      # 🔥 Bigger text
    thickness = 3         # 🔥 Thicker text
    box_thickness = 3     # 🔥 Thicker boxes

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                cls = int(cls)
                class_name = model.model.names.get(cls, str(cls))  # 🔥 get class name

                cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

                cv2.putText(
                    img,
                    f"{class_name}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

    return img

# --- Evaluation - Score all predictions by quality ---
all_predictions = []

print("Evaluating all validation images...")

for img_path in val_images:
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    gt_boxes = read_gt_boxes(label_path, w, h)

    results = model.predict(source=img_path, conf=0.25, iou=0.45, verbose=False)
    preds = results[0].boxes

    pred_boxes = []
    pred_confs = []
    for b in preds:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        pred_boxes.append([cls, x1, y1, x2, y2])
        pred_confs.append(conf)

    iou_threshold = 0.5
    used_pred = set()
    correct_matches = 0
    total_gt = len(gt_boxes)
    class_errors = 0

    for gt in gt_boxes:
        gt_cls, x1, y1, x2, y2 = gt
        matched = False

        for i, pred in enumerate(pred_boxes):
            pred_cls, px1, py1, px2, py2 = pred
            iou = compute_iou([x1, y1, x2, y2], [px1, py1, px2, py2])

            if iou >= iou_threshold:
                used_pred.add(i)
                if int(pred_cls) == int(gt_cls):
                    correct_matches += 1
                else:
                    class_errors += 1
                matched = True
                break

    false_positives = len(pred_boxes) - len(used_pred)
    missed_gt = total_gt - len(used_pred)

    # Calculate quality score (higher is better)
    # Perfect match = 1.0, worse predictions get lower scores
    if total_gt > 0:
        recall = correct_matches / total_gt
    else:
        recall = 1.0
    
    precision = correct_matches / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    
    # F1 score
    if recall + precision > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    avg_conf = np.mean(pred_confs) if len(pred_confs) > 0 else 0.0
    
    # Combined quality score: F1 * average confidence
    quality_score = f1_score * avg_conf

    all_predictions.append({
        'img_path': img_path,
        'label_path': label_path,
        'quality_score': quality_score,
        'f1_score': f1_score,
        'avg_conf': avg_conf,
        'correct_matches': correct_matches,
        'total_gt': total_gt,
        'class_errors': class_errors,
        'false_positives': false_positives,
        'missed_gt': missed_gt,
        'results': results
    })

# Sort by quality score (descending) and take top 10
all_predictions.sort(key=lambda x: x['quality_score'], reverse=True)
top_10 = all_predictions[:10]

print(f"\nEvaluated {len(all_predictions)} images")
print(f"Saving top 10 with highest quality scores...\n")

# --- Visualize top 10 ---
output_dir = "./best_quality_predictions"
os.makedirs(output_dir, exist_ok=True)

for idx, pred_info in enumerate(top_10, 1):
    img_path = pred_info['img_path']
    label_path = pred_info['label_path']
    results = pred_info['results']
    
    filename = os.path.basename(img_path)
    print(f"#{idx}: {filename}")
    print(f"    Quality: {pred_info['quality_score']:.4f} | F1: {pred_info['f1_score']:.4f} | Avg Conf: {pred_info['avg_conf']:.4f}")
    print(f"    Correct: {pred_info['correct_matches']}/{pred_info['total_gt']} | "
          f"Class Errors: {pred_info['class_errors']} | "
          f"False Positives: {pred_info['false_positives']} | "
          f"Missed: {pred_info['missed_gt']}")

    # --- Visualization ---
    gt_img = draw_yolo_boxes(img_path, label_path, color=(0, 255, 0), label_type="GT")

    pred_img = results[0].plot(
        line_width=3,
        font_size=20
    )

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Ground Truth (Green)")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"Prediction - Quality: {pred_info['quality_score']:.4f}")
    axs[1].axis("off")

    out_path = os.path.join(output_dir, f"top_{idx}_quality_{pred_info['quality_score']:.4f}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

print(f"\nTop 10 best quality predictions saved to: {output_dir}")