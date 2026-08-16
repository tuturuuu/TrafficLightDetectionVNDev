from ultralytics import YOLO
import torch
import sys
import os

# Đăng ký các module tự chế (SE, CBAM) vào Ultralytics
sys.path.append(os.path.abspath('./utils'))
import ultralytics.nn.tasks as tasks
from custom_modules import CBAM, SE
tasks.CBAM = CBAM
tasks.SE = SE

def main():
    print("🚀 Bắt đầu quá trình huấn luyện TILING với YOLO26m trên SeaDroneSee Dataset")
    print("Kiểm tra CUDA (GPU):", torch.cuda.is_available())

    # Khởi tạo mô hình (Dùng kiến trúc YOLO26 custom)
    model = YOLO('yolo_config/yolo26m-modified.yaml') 
    
    # Bắt đầu huấn luyện
    results = model.train(
        data='./sds_data_tiled.yaml',
        epochs=100,             
        imgsz=640,              
        batch=8, # Chuyển về batch 8 để tránh OOM     
        name='sds_yolo26m_tiling',  
        project='./runs',       
        device=0,               
        workers=8               
    )

    # Đánh giá sau khi train
    metrics = model.val()
    print("✅ Hoàn tất! Kết quả đánh giá:", metrics)

if __name__ == '__main__':
    main()
