# legacy/

Code cũ, **không dùng cho thí nghiệm mới**. Giữ lại để tra cứu và để
`test/` so sánh pipeline mới với kết quả cũ. Các file ở đây đã được sửa lại
đường dẫn tương đối sau khi chuyển vào `legacy/`, ngoài ra giữ nguyên.
Chạy từ thư mục gốc repo, vd `python legacy/dense_tiling_experiment.py ...`.

| File / thư mục | Trước là | Thay bằng | Ghi chú |
|---|---|---|---|
| `dense_tiling_experiment.py` | Thí nghiệm chính uniform vs GridNet | `scripts/evaluate.py` + `configs/eval/*.yaml` | **Chuẩn so sánh** của `test/test_parity.py` và `test/test_units.py`; đừng sửa logic. Eval của file này không xét class |
| `evaluations/` | Eval pipeline CNN thế hệ 1, vẽ ảnh, đếm tham số | `scripts/evaluate.py` (`tile_selector.type: tile_cnn`) | `evaluation_with_cnn.py` là chuẩn so sánh cho selector `tile_cnn` |
| `train_seadronesee_cbam.py`, `train_seadronesee_tiling_cbam.py` | Train YOLOv8-CBAM trên SeaDroneSee | `scripts/train_detector.py` + `configs/train/detector_seadronesee.yaml` | |
| `kayuan2024/`, `bosch/` | Script train phase 1 (export từ Colab) | `scripts/train_detector.py` | Đường dẫn hardcode `/home/vietpham/...`, không chạy được nguyên trạng |
| `cnn_classifier/` | CNN chọn tile thế hệ 1, Grad-CAM (`small_cnn_heatmap.py`), đặc trưng thủ công | selector `tile_cnn` trong `scripts/evaluate.py` | `small_cnn.py` vẫn được `scripts/train_selector.py` gọi khi `type: tile_cnn` |
| `custom_modules.py` | `utils/custom_modules.py` | `src/adaptile/models/attention.py` | Chỉ import lại CBAM/SE, để script cũ chạy được |
| `adaptive_tiling_deprecated.py` | Adaptive tiling theo mật độ vật thể (hướng #1) | (bỏ hướng này) | Deprecated |
