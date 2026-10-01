# weights/

Tất cả weight của repo. Script mặc định ghi weight mới vào đây (vd `grid_proposal_net.py train --out weights/...`).

Quy ước tên: `<model>_<dataset>_<cấu hình>.pth`. Khi thêm weight mới, bổ sung một dòng vào bảng dưới.

| File | Model | Dataset | Ngày | Ghi chú |
|---|---|---|---|---|
| `gridnet_kayuan_8x8.pth` | GridNet 8×8 | Kayuan (`data/kayuan`, train split) | 2026-09-19 | `--pos-weight 10`, early stop ở epoch 34/40. Log: `eval_logs/kayuan_gridnet_train.log` |
| `gridnet_seadronesee_8x8.pth` | GridNet 8×8 | SeaDroneSee (`data/SeaDroneSees_70_15_15`, suy ra từ tên cũ) | 2026-09-10 | Tên cũ `grid_net_70_15_15.pth` |
| `gridnet_unknown_20260817.pth` | GridNet 8×8 | ❓ **Chưa rõ** | 2026-08-17 | Tên cũ `grid_net.pth`. Cùng thời điểm với `outputs/kayuan/scaling_results_20260817.json` (8 tile/ảnh ở tile 640, tức ảnh 1920×1080 giống Kayuan) nhưng chưa xác nhận |
| `tile_proposal_cnn_model.pth` | CNN chọn tile (thế hệ 1) | ❓ Chưa rõ | 2026-08-17 | Trước ở `cnn_classifier/`. **Mặc định** của `evaluations/*` và `cnn_classifier/*` |
| `tile_proposal_cnn_model_rootcopy.pth` | CNN chọn tile (thế hệ 1) | ❓ Chưa rõ | 2026-08-17 | Trước ở thư mục gốc. Cùng tên với file trên nhưng **nội dung khác**, không script nào dùng |
| `tile_proposal_cnn_kayuan.pth` | CNN chọn tile (thế hệ 1) | Kayuan (suy ra từ tên) | 2026-10-01 | Chưa có log eval |
| `yolo26n.pt` | YOLO26n | COCO | 2026-08-18 | Có vẻ là checkpoint pretrained gốc của Ultralytics |

Weight detector đã train (YOLOv8-CBAM, …) **không** nằm ở đây mà ở `runs/<tên_run>/weights/best.pt`.
