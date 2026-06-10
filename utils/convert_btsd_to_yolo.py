from pathlib import Path
import os
import yaml
import json
import random
import shutil
import cv2
import matplotlib.pyplot as plt

# Update these paths according to your setup
BSTLD_ROOT = "/home/vietpham/dataset/bstld-DatasetNinja"  # Update this!
OUTPUT_ROOT = "./bstld_yolo_format"  # Update this!
RESULTS_DIR = "./bosch_result/results"  # Update this!


# BSTLD class mapping
CLASS_NAMES = ['red', 'yellow', 'green', 'off']

# Map all BSTLD label variants to the 4 main classes
CLASS_MAPPING = {
    # Lowercase versions
    'red': 0,
    'yellow': 1, 
    'green': 2,
    'off': 3,
    # Capitalized versions
    'Red': 0,
    'Yellow': 1,
    'Green': 2,
    'Off': 3,
    # Red variants
    'RedLeft': 0,
    'RedRight': 0,
    'RedStraight': 0,
    'RedStraightLeft': 0,
    'RedStraightRight': 0,
    # Yellow variants
    'YellowLeft': 1,
    'YellowRight': 1,
    'YellowStraight': 1,
    'YellowStraightLeft': 1,
    'YellowStraightRight': 1,
    # Green variants
    'GreenLeft': 2,
    'GreenRight': 2,
    'GreenStraight': 2,
    'GreenStraightLeft': 2,
    'GreenStraightRight': 2,
}


def debug_save_image_with_boxes(img_path, yolo_label_path, save_path):
    """Plot image with YOLO boxes and save to disk (server-friendly)."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Read YOLO labels
    if not os.path.exists(yolo_label_path):
        print(f"No labels found: {yolo_label_path}")
        return

    boxes = []
    with open(yolo_label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, bw, bh = map(float, parts)
            # Convert normalized → pixel coords
            xc *= w
            yc *= h
            bw *= w
            bh *= h
            x1 = int(xc - bw/2)
            y1 = int(yc - bh/2)
            x2 = int(xc + bw/2)
            y2 = int(yc + bh/2)
            boxes.append((class_id, x1, y1, x2, y2))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")

    for class_id, x1, y1, x2, y2 in boxes:
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        )
        ax.text(x1, y1 - 3, CLASS_NAMES[int(class_id)], fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved debug image to: {save_path}")


def create_data_yaml(output_root):
    """Create YOLO data.yaml configuration file"""
    data_yaml = {
        'path': os.path.abspath(output_root),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    yaml_path = os.path.join(output_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml at: {yaml_path}")


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bbox format to YOLO format
    Input: [x_min, y_min, x_max, y_max] or dict with x_min, y_min, x_max, y_max
    YOLO format: [x_center, y_center, width, height] normalized [0-1]
    """
    if isinstance(bbox, dict):
        x_min = bbox['x_min']
        y_min = bbox['y_min']
        x_max = bbox['x_max']
        y_max = bbox['y_max']
    else:
        x_min, y_min, x_max, y_max = bbox
    
    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def load_annotations_from_json(json_path):
    """Load annotations from JSON file (DatasetNinja format)"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def process_split_new_structure(split_name, bstld_root, output_root):
    """
    Process a split with the new BSTLD structure:
    - Images in: {split}/img/
    - Annotations in: {split}/ann/ (JSON files with same name as images)
    """
    
    processed_count = 0
    skipped_count = 0
    total_boxes = 0
    class_counts = {label: 0 for label in CLASS_NAMES}
    
    # Paths for this split
    img_dir = os.path.join(bstld_root, split_name, 'img')
    ann_dir = os.path.join(bstld_root, split_name, 'ann')
    
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found: {img_dir}")
        return
    
    if not os.path.exists(ann_dir):
        print(f"Warning: Annotation directory not found: {ann_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(Path(img_dir).glob(ext))
    
    print(f"  Found {len(image_files)} images in {img_dir}")
    
    for img_path in image_files:
        img_filename = img_path.name
        img_name = img_path.stem
        
        # Look for corresponding annotation file
        # Annotation files are named {filename}.png.json (includes the extension)
        ann_path = os.path.join(ann_dir, f"{img_filename}.json")
        
        if not os.path.exists(ann_path):
            print(f"  Warning: No annotation file for {img_filename}")
            skipped_count += 1
            continue
        
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Cannot read image: {img_path}")
            skipped_count += 1
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Copy image to output
        output_img_path = os.path.join(output_root, 'images', split_name, img_filename)
        shutil.copy2(str(img_path), output_img_path)
        
        # Load annotations
        try:
            ann_data = load_annotations_from_json(ann_path)
        except Exception as e:
            print(f"  Warning: Error loading annotations from {ann_path}: {e}")
            skipped_count += 1
            continue
        
        # Parse annotations - DatasetNinja format
        yolo_annotations = []
        
        # Check different possible structures
        objects = []
        if isinstance(ann_data, dict):
            if 'objects' in ann_data:
                objects = ann_data['objects']
            elif 'annotations' in ann_data:
                objects = ann_data['annotations']
            elif 'boxes' in ann_data:
                objects = ann_data['boxes']
        elif isinstance(ann_data, list):
            objects = ann_data
        
        for obj in objects:
            # Extract label
            label = None
            for key in ['classTitle', 'class', 'label', 'category']:
                if key in obj:
                    label = obj[key]
                    break
            
            if label is None:
                continue
            
            # Map label to class
            label = str(label)
            if label not in CLASS_MAPPING:
                # Try to match by color prefix
                label_lower = label.lower()
                if 'red' in label_lower:
                    class_id = 0
                elif 'yellow' in label_lower:
                    class_id = 1
                elif 'green' in label_lower:
                    class_id = 2
                elif 'off' in label_lower:
                    class_id = 3
                else:
                    print(f"  Warning: Unknown label '{label}' in {img_filename}, skipping")
                    continue
            else:
                class_id = CLASS_MAPPING[label]
            
            # Extract bbox - check different formats
            bbox = None
            
            # Format 1: points array [[x1,y1], [x2,y2]]
            if 'points' in obj and 'exterior' in obj['points']:
                points = obj['points']['exterior']
                if len(points) == 2:
                    x_min = min(points[0][0], points[1][0])
                    y_min = min(points[0][1], points[1][1])
                    x_max = max(points[0][0], points[1][0])
                    y_max = max(points[0][1], points[1][1])
                    bbox = [x_min, y_min, x_max, y_max]
            
            # Format 2: direct coordinates
            elif all(k in obj for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                bbox = [obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']]
            
            # Format 3: bbox key
            elif 'bbox' in obj:
                bbox_data = obj['bbox']
                if isinstance(bbox_data, list) and len(bbox_data) == 4:
                    bbox = bbox_data
                elif isinstance(bbox_data, dict):
                    bbox = [bbox_data['x_min'], bbox_data['y_min'], 
                           bbox_data['x_max'], bbox_data['y_max']]
            
            if bbox is None:
                print(f"  Warning: Could not extract bbox from {img_filename}")
                continue
            
            x_min, y_min, x_max, y_max = bbox
            
            # Validate coordinates
            if x_max <= x_min or y_max <= y_min:
                print(f"  Warning: Invalid box coordinates in {img_filename}")
                continue
            
            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox, img_width, img_height
            )
            
            # Clamp values to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            total_boxes += 1
            class_counts[CLASS_NAMES[class_id]] += 1
        
        # Write YOLO label file
        output_label_path = os.path.join(output_root, 'labels', split_name, f"{img_name}.txt")
        with open(output_label_path, 'w') as f:
            if yolo_annotations:
                f.write('\n'.join(yolo_annotations))
        
        processed_count += 1
    
    print(f"\n  {split_name} Summary:")
    print(f"    Processed: {processed_count} images")
    print(f"    Skipped: {skipped_count} images")
    print(f"    Total boxes: {total_boxes}")
    if total_boxes > 0:
        print(f"    Class distribution:")
        for label in CLASS_NAMES:
            count = class_counts[label]
            percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
            print(f"      {label}: {count} ({percentage:.1f}%)")


def convert_bstld_to_yolo(bstld_root, output_root):
    """
    Convert BSTLD dataset to YOLO format
    Works with DatasetNinja structure: {train,test}/{img,ann}
    """
    print("Converting BSTLD to YOLO format...")
    print(f"Input: {bstld_root}")
    print(f"Output: {output_root}")
    
    if os.path.exists(output_root):
        print(f"Clearing previous output: {output_root}")
        shutil.rmtree(output_root)
    
    # Create fresh output directories
    for split in ['train', 'val', 'test']:
        Path(output_root, 'images', split).mkdir(parents=True, exist_ok=True)
        Path(output_root, 'labels', split).mkdir(parents=True, exist_ok=True)
        
    # Create output directories
    for split in ['train', 'val', 'test']:
        Path(output_root, 'images', split).mkdir(parents=True, exist_ok=True)
        Path(output_root, 'labels', split).mkdir(parents=True, exist_ok=True)
    
    # Process train set - split into train/val
    print("\nProcessing train set...")
    train_img_dir = os.path.join(bstld_root, 'train', 'img')
    if os.path.exists(train_img_dir):
        # Get all images
        train_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            train_images.extend(list(Path(train_img_dir).glob(ext)))
        
        # Shuffle and split
        random.shuffle(train_images)
        split_idx = int(0.8 * len(train_images))
        train_subset = train_images[:split_idx]
        val_subset = train_images[split_idx:]
        
        print(f"  Split {len(train_images)} images into:")
        print(f"    Train: {len(train_subset)} images")
        print(f"    Val: {len(val_subset)} images")
        
        # Process train subset
        print("\n  Processing training subset...")
        process_images_subset(train_subset, bstld_root, output_root, 'train', 'train')
        
        # Process val subset
        print("\n  Processing validation subset...")
        process_images_subset(val_subset, bstld_root, output_root, 'train', 'val')
    
    # Process test set
    print("\nProcessing test set...")
    test_img_dir = os.path.join(bstld_root, 'test', 'img')
    if os.path.exists(test_img_dir):
        process_split_new_structure('test', bstld_root, output_root)
    
    # Create data.yaml
    create_data_yaml(output_root)
    
    print(f"\nConversion complete! YOLO dataset saved to: {output_root}")


def process_images_subset(image_paths, bstld_root, output_root, source_split, target_split):
    """Process a subset of images (for train/val split)"""
    
    processed_count = 0
    skipped_count = 0
    total_boxes = 0
    class_counts = {label: 0 for label in CLASS_NAMES}
    
    ann_dir = os.path.join(bstld_root, source_split, 'ann')
    
    for img_path in image_paths:
        img_filename = img_path.name
        img_name = img_path.stem
        
        # Look for corresponding annotation file
        # Annotation files are named {filename}.png.json (includes the extension)
        ann_path = os.path.join(ann_dir, f"{img_filename}.json")
        
        if not os.path.exists(ann_path):
            skipped_count += 1
            continue
        
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_count += 1
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Copy image to output
        output_img_path = os.path.join(output_root, 'images', target_split, img_filename)
        shutil.copy2(str(img_path), output_img_path)
        
        # Load annotations
        try:
            ann_data = load_annotations_from_json(ann_path)
        except Exception as e:
            skipped_count += 1
            continue
        
        # Parse annotations
        yolo_annotations = []
        
        objects = []
        if isinstance(ann_data, dict):
            if 'objects' in ann_data:
                objects = ann_data['objects']
            elif 'annotations' in ann_data:
                objects = ann_data['annotations']
        elif isinstance(ann_data, list):
            objects = ann_data
        
        for obj in objects:
            # Extract label
            label = None
            for key in ['classTitle', 'class', 'label', 'category']:
                if key in obj:
                    label = obj[key]
                    break
            
            if label is None:
                continue
            
            label = str(label)
            if label not in CLASS_MAPPING:
                label_lower = label.lower()
                if 'red' in label_lower:
                    class_id = 0
                elif 'yellow' in label_lower:
                    class_id = 1
                elif 'green' in label_lower:
                    class_id = 2
                elif 'off' in label_lower:
                    class_id = 3
                else:
                    continue
            else:
                class_id = CLASS_MAPPING[label]
            
            # Extract bbox
            bbox = None
            if 'points' in obj and 'exterior' in obj['points']:
                points = obj['points']['exterior']
                if len(points) == 2:
                    x_min = min(points[0][0], points[1][0])
                    y_min = min(points[0][1], points[1][1])
                    x_max = max(points[0][0], points[1][0])
                    y_max = max(points[0][1], points[1][1])
                    bbox = [x_min, y_min, x_max, y_max]
            elif all(k in obj for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                bbox = [obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']]
            
            if bbox is None:
                continue
            
            x_min, y_min, x_max, y_max = bbox
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox, img_width, img_height
            )
            
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            total_boxes += 1
            class_counts[CLASS_NAMES[class_id]] += 1
        
        # Write YOLO label file
        output_label_path = os.path.join(output_root, 'labels', target_split, f"{img_name}.txt")
        with open(output_label_path, 'w') as f:
            if yolo_annotations:
                f.write('\n'.join(yolo_annotations))
        
        processed_count += 1
    
    print(f"    Processed: {processed_count} images")
    print(f"    Skipped: {skipped_count} images")
    print(f"    Total boxes: {total_boxes}")


if __name__ == "__main__":
    print("Converting BSTLD to YOLO format...")
    convert_bstld_to_yolo(BSTLD_ROOT, OUTPUT_ROOT)