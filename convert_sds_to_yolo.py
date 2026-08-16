import os
import json
from pathlib import Path
from tqdm import tqdm

def convert_coco_json_to_yolo(json_path, out_labels_dir):
    print(f"Đang xử lý file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if 'images' not in data or 'annotations' not in data:
        print("⚠️ File JSON không đúng chuẩn COCO. Bỏ qua.")
        return

    # Tạo mapping image_id -> image_info
    images = {img['id']: img for img in data['images']}
    
    # Tạo mapping category_id -> yolo_class_id (bắt đầu từ 0)
    cat_ids = sorted([cat['id'] for cat in data.get('categories', [])])
    cat_mapping = {c_id: i for i, c_id in enumerate(cat_ids)}
    
    print("Danh sách classes tự động nhận diện:")
    for cat in data.get('categories', []):
        print(f" - ID YOLO: {cat_mapping[cat['id']]} -> {cat['name']}")
        
    os.makedirs(out_labels_dir, exist_ok=True)
    
    # Gom annotations theo image_id
    from collections import defaultdict
    annos_by_img = defaultdict(list)
    for ann in data['annotations']:
        annos_by_img[ann['image_id']].append(ann)
        
    # Tạo file txt cho từng ảnh
    count = 0
    for img_id, img_info in tqdm(images.items(), desc="Đang chuyển đổi"):
        file_name = img_info['file_name']
        base_name = Path(file_name).stem
        txt_path = os.path.join(out_labels_dir, f"{base_name}.txt")
        
        img_w = img_info['width']
        img_h = img_info['height']
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            if img_id in annos_by_img:
                for ann in annos_by_img[img_id]:
                    # Bbox chuẩn COCO: [x_min, y_min, width, height]
                    x_min, y_min, w, h = ann['bbox']
                    
                    # Công thức chuyển sang YOLO (chuẩn hóa về [0, 1])
                    x_center = (x_min + w / 2) / img_w
                    y_center = (y_min + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    
                    # Đảm bảo không bị tràn viền (1.0001 -> 1.0)
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))
                    
                    cls_id = cat_mapping[ann['category_id']]
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        count += 1
    print(f"✅ Đã tạo {count} file nhãn tại {out_labels_dir}\n")

if __name__ == "__main__":
    # Đường dẫn thư mục của bạn trên máy
    base_dir = "./sds_dataset/compressed"
    anno_dir = os.path.join(base_dir, "annotations")
    
    if not os.path.exists(anno_dir):
        print(f"❌ Không tìm thấy thư mục annotations tại: {anno_dir}")
        exit(1)
        
    for json_file in os.listdir(anno_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(anno_dir, json_file)
            
            # Phân loại đây là train hay val dựa vào tên file
            if "train" in json_file.lower():
                split_name = "train"
            elif "val" in json_file.lower():
                split_name = "val"
            elif "test" in json_file.lower():
                split_name = "test"
            else:
                split_name = Path(json_file).stem # Lấy tên file nếu không đoán được
            
            # YOLO tự động tìm nhãn trong thư mục 'labels' song song với 'images'
            out_labels_dir = os.path.join(base_dir, "labels", split_name)
            
            convert_coco_json_to_yolo(json_path, out_labels_dir)
            
    print("🚀 HOÀN TẤT! Cấu trúc thư mục labels chuẩn YOLO đã sẵn sàng.")
