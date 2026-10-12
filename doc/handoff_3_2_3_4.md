# Sửa các mục 3.2–3.4 (2026-10-12)

## Split theo nguồn video

Kayuan có **73** chuỗi timestamp giao nhau giữa mỗi cặp tập, thay vì 68 ghi trong HANDOFF cũ. Định danh chuỗi là 14 chữ số đầu của tên file sau khi bỏ `_` (ngày + giờ bắt đầu video). Split mới seed 42 giữ nguyên cả chuỗi, phân bổ theo số ảnh khoảng 70/15/15:

| Dataset | Train | Val | Test | Nhóm trùng giữa các tập |
|---|---:|---:|---:|---:|
| Kayuan | 14127 | 3036 | 3035 | 0 |
| SeaDroneSee | 7312 | 1579 | 1586 | 0 |

Kayuan loại 51 ảnh thiếu file nhãn, có danh sách trong `kayuan_split_audit.json`. Không tự tạo nhãn rỗng. SeaDroneSee cũ có 38 nhóm nguồn giao nhau giữa mỗi cặp tập. Dùng `source.drone/folder_name/video` từ COCO gốc; ảnh chụp tĩnh được nhóm bảo thủ theo drone + folder. Kiểm tra này chứng minh không trùng định danh nguồn, không chứng minh các video khác nhau không chụp cùng địa điểm.

```bash
python scripts/data/check_split_leakage.py data/kayuan --kind kayuan --exclude-missing-labels --output-root data/kayuan_sequence --report doc/kayuan_split_audit.json
python scripts/data/check_split_leakage.py data/SeaDroneSees_70_15_15 --kind seadrone --metadata-root data/SeaDroneSees --output-root data/seadrone_sequence --report doc/seadrone_split_audit.json
```

Script từ chối output đã tồn tại. Dữ liệu mới dùng symlink tới bản gốc, có `data.yaml`, manifest và audit. Trên máy khác phải tạo lại dữ liệu; không di chuyển/xóa bản gốc. Các config train/eval đã dùng đường dẫn mới. GridNet được truyền val riêng, thay cho chia ngẫu nhiên 10% frame train.

**Weight detector/selector cũ từng thấy frame của test mới. Phải train lại trên split mới trước khi báo cáo độ chính xác.** Config eval vẫn tham chiếu weight cũ để kiểm tra pipeline; không dùng kết quả đó làm số liệu nghiên cứu. Tile train/val cũng phải sinh mới dưới root mới.

## Kiến trúc YOLO26

`yolo26m+none` dùng `configs/models/yolo26m-p2.yaml`. SE/CBAM dùng các file tương ứng, cùng đầu ra P2/P3/P4/P5. Detect của SE/CBAM lấy `[22,25,28,31]`; Concat cuối lấy C2PSA (13). Skip P3/P4 giữ lựa chọn trước attention (4/7), đúng cách nối có sẵn; không tự thay quyết định nghiên cứu còn mở. File `yolo26m-modified.yaml` được sửa giống SE để lệnh cũ vẫn chạy. Script eval2 dùng đường dẫn P2 chính xác.

## Đo hiệu năng

JSON eval thêm:

- `latency_ms`: p50/p95 cho select/detect/merge/total.
- `per_image_latency_ms`: từng mẫu theo thứ tự ảnh đã sort.
- `merge_s`: tổng thời gian merge.
- `model_complexity`: params, GFLOPs, input_size của detector và selector đang chạy. THOP đếm các toán tử hỗ trợ, quy ước GFLOPs = 2 × MACs, một ảnh/crop, không phải cả tổng tile trên ảnh. Profile trên bản sao CPU ở kích thước đầu vào thực tế; lỗi/thiếu profiler trả null kèm lý do.
- `peak_gpu_memory_mb`: peak bộ nhớ tensor allocated theo device, reset sau warmup cho mỗi run. CPU trả `{}`; không phải bộ nhớ reserved hay tổng số trong nvidia-smi.

Đọc ảnh và warmup nằm ngoài thời gian đo. Detect bao gồm crop và chuyển prediction về CPU; total có cả tạo tile và overhead. GPU được synchronize tại các mốc đo. Hai GPU đang có job khác, nên xác minh đầu cuối chạy trên CPU; chưa thu benchmark GPU độc lập.

## Kiểm chứng

167 test đạt, 3 test train đầy đủ được skip. Cả P2/SE/CBAM chạy forward CPU thành công. Eval thật trên hai ảnh Kayuan, CPU, input 64 xuất JSON có latency từng ảnh, p50/p95, params/GFLOPs; kiểm tra lại trực tiếp hai split mới không có nhóm nguồn giao nhau. `git diff --check` sạch.
