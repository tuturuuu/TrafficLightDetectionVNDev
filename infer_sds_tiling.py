import os
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# Đăng ký custom modules để model có thể load được
sys.path.append(os.path.abspath('./utils'))
from custom_modules import CBAM, SE
tasks.CBAM = CBAM
tasks.SE = SE

# ========== CẤU HÌNH ==========
# Đường dẫn đến file weights tốt nhất sau khi train xong
MODEL_PATH = "runs/detect/sds_yolo26m_baseline/weights/best.pt" 
# Thư mục ảnh cần test (mặc định lấy tập val của SeaDroneSee)
TEST_IMAGES_DIR = "/home/anhld/lab_model/TrafficLightDetectionVNDev/sds_dataset/compressed/images/val"
# Thư mục lưu kết quả ảnh đã vẽ khung
OUTPUT_DIR = "sds_tiled_results"

TILE_SIZE = 640
OVERLAP = 0.2       # Chồng lấp 20% giữa các ô cắt (để không bị đứt đôi vật thể ở mép)
CONF_THRES = 0.25   # Chỉ lấy dự đoán có độ tự tin > 25%
IOU_THRES = 0.45    # Mức độ giao nhau để gộp các box trùng lặp

# Từ điển ánh xạ ID sang tên class
CLASS_NAMES = {
    0: "ignored", 1: "swimmer", 2: "boat", 
    3: "jetski", 4: "life_saving_appliances", 5: "buoy"
}

def get_tile_starts(length, tile_size, overlap):
    step = int(tile_size * (1.0 - overlap))
    starts = list(range(0, length - tile_size + 1, step))
    last_valid = length - tile_size
    if not starts or starts[-1] < last_valid:
        starts.append(last_valid)
    return starts

def run_tiled_inference(image_path, model):
    img = cv2.imread(image_path)
    if img is None: return None
    
    h, w = img.shape[:2]
    
    # 1. CẮT ẢNH THÀNH CÁC Ô (TILES)
    tiles = []
    starts_y = get_tile_starts(h, TILE_SIZE, OVERLAP)
    starts_x = get_tile_starts(w, TILE_SIZE, OVERLAP)
    
    for ty in starts_y:
        for tx in starts_x:
            tile = img[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]
            
            # Đệm viền màu xám nếu ô cắt bị hụt ở phần rìa phải/dưới của ảnh
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                tile = cv2.copyMakeBorder(
                    tile, 0, TILE_SIZE - tile.shape[0], 0, TILE_SIZE - tile.shape[1],
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
            tiles.append({"image": tile, "x": tx, "y": ty})

    if not tiles: return img
    
    # 2. CHẠY YOLO TRÊN TOÀN BỘ CÁC Ô
    tile_images = [t["image"] for t in tiles]
    # Truyền cả mảng ảnh ô cắt vào YOLO để tăng tốc độ xử lý (Batch Inference)
    results = model.predict(tile_images, conf=CONF_THRES, imgsz=TILE_SIZE, verbose=False)
    
    boxes_list = []
    scores_list = []
    class_list = []
    
    # 3. GOM KẾT QUẢ VÀ CỘNG NGƯỢC TỌA ĐỘ VỀ ẢNH GỐC
    for tile_info, res in zip(tiles, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
            
        for box, conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            # box = [x1, y1, x2, y2] là tọa độ tính trong nội bộ ô cắt
            # Cộng thêm tọa độ (x, y) góc trên cùng bên trái của ô cắt để trả về hệ quy chiếu ảnh gốc
            gx1 = box[0] + tile_info["x"]
            gy1 = box[1] + tile_info["y"]
            gx2 = box[2] + tile_info["x"]
            gy2 = box[3] + tile_info["y"]
            
            # Hàm NMSBoxes của OpenCV yêu cầu định dạng [x_min, y_min, width, height]
            boxes_list.append([gx1, gy1, gx2 - gx1, gy2 - gy1]) 
            scores_list.append(float(conf))
            class_list.append(int(cls))

    # 4. LỌC NMS (Non-Maximum Suppression) 
    # Xóa các box trùng lặp ở phần bị cắt chồng lên nhau giữa 2 ô
    final_boxes = []
    if len(boxes_list) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, CONF_THRES, IOU_THRES)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes_list[i]
                final_boxes.append({
                    "box": [int(x), int(y), int(x+bw), int(y+bh)],
                    "conf": scores_list[i],
                    "cls": class_list[i]
                })
                
    # 5. VẼ KẾT QUẢ LÊN ẢNH GỐC ĐỂ XUẤT RA
    out_img = img.copy()
    for det in final_boxes:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["cls"]
        conf = det["conf"]
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
        
        # Tạo màu xanh/đỏ/tím ngẫu nhiên nhưng cố định cho từng ID class
        color = ((cls_id * 50) % 255, (cls_id * 120 + 50) % 255, (cls_id * 80 + 150) % 255)
        
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return out_img

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại {MODEL_PATH}.")
        print("Vui lòng sửa biến MODEL_PATH trong code cho đúng tên thư mục baseline bạn vừa train xong nhé!")
        return

    print("⏳ Đang tải mô hình YOLO đã huấn luyện...")
    model = YOLO(MODEL_PATH)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Lấy tất cả ảnh từ tập val
    img_files = list(Path(TEST_IMAGES_DIR).glob("*.jpg")) + list(Path(TEST_IMAGES_DIR).glob("*.png"))
    if not img_files:
        print(f"❌ Không tìm thấy ảnh nào trong {TEST_IMAGES_DIR}")
        return
        
    # Lấy thử 15 ảnh đầu tiên để test
    test_files = img_files[:15]
    
    print(f"🚀 Bắt đầu nhận diện bằng kỹ thuật Tiling cho {len(test_files)} ảnh...")
    for img_path in test_files:
        print(f" - Đang phân tích ảnh: {img_path.name}")
        result_img = run_tiled_inference(str(img_path), model)
        
        if result_img is not None:
            out_path = os.path.join(OUTPUT_DIR, f"tiled_{img_path.name}")
            cv2.imwrite(out_path, result_img)
            
    print(f"\n✅ Hoàn tất! Hãy mở thư mục '{OUTPUT_DIR}' trên VS Code (hoặc tải về) để xem tận mắt kết quả.")

if __name__ == "__main__":
    main()
