# Adaptive Tiling: chọn tile thông minh để tăng tốc YOLO + tiling

Repo này là **phase 2** của paper [AT-YOLO: enhanced YOLO-based detector with attention mechanism](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14309/143090V/AT-YOLO--enhanced-YOLO-based-detector-with-attention-mechanism/10.1117/12.3120875.full) (SPIE Proc. 14309, 143090V, DOI: [10.1117/12.3120875](https://doi.org/10.1117/12.3120875)).

> **Người mới:** đọc lần lượt [Bối cảnh](#1-bối-cảnh) → [Phương pháp](#2-phương-pháp-gridnet) → [Cấu trúc repo](#3-cấu-trúc-repo) → [Chạy thử](#5-chạy-thử). Trước khi chạy bất cứ thứ gì, đọc [Lưu ý quan trọng](#7-lưu-ý-quan-trọng).

---

## 1. Bối cảnh

**Phase 1 (AT-YOLO):** YOLO + **tiling** (cắt ảnh lớn thành nhiều tile để giữ độ phân giải của vật thể nhỏ) + **attention (CBAM/SE)** cho kết quả phát hiện vật thể nhỏ (đèn giao thông) tốt hơn đáng kể so với YOLO chạy trên ảnh nguyên.

**Vấn đề:** tiling rất tốn thời gian. YOLO phải chạy trên *mỗi tile*, nên ảnh cắt thành N tile thì chạy chậm khoảng N lần, trong khi phần lớn tile là nền trống (trời, đường, mặt biển) và không chứa vật thể nào.

**Mục tiêu phase 2:** giảm số tile phải đưa qua YOLO mà **không làm giảm độ chính xác** so với tiling đầy đủ (uniform tiling).

> GridNet **không làm tăng** recall hay mAP. Nó chỉ bỏ bớt tile, nên trường hợp tốt nhất là độ chính xác bằng uniform tiling. Đóng góp của phase 2 là **tốc độ ở cùng mức độ chính xác**.

### Các hướng đã thử

| # | Hướng | Code | Trạng thái | Ghi chú |
|---|---|---|---|---|
| 1 | Adaptive tiling theo mật độ vật thể (NBA) | `tiling/adaptive_tiling_deprecated.py` | ❌ **Deprecated** | Không nghiên cứu tiếp |
| 2 | CNN phân loại từng tile (sau khi cắt) | `cnn_classifier/` | ⚠️ Thay thế bởi #3 | Chọn tile ổn, nhưng nghẽn ở khâu batching vì phải chạy CNN trên từng tile, nên chi phí tăng theo số tile |
| 3 | **GridNet**: một lượt chạy trên cả ảnh | `grid_proposal_net.py` | ✅ **Hướng chính** | Bản cải tiến của #2, không bị nghẽn batching, chi phí cố định mỗi ảnh |

---

## 2. Phương pháp: GridNet

```
                      ┌───────────────────────────┐
 Ảnh gốc ──resize──▶  │ GridNet (~49K params)     │ ──▶ heatmap 8×8 (xác suất có vật thể)
 (vd 1920×1080) 256²  │ 1 forward pass / ảnh      │            │ threshold
                      └───────────────────────────┘            ▼
                                                    mask ô "giữ" trên lưới 8×8
 Ảnh gốc ──cắt tile (uniform, overlap 20%)──▶ tile ──▶ giữ tile chạm ít nhất 1 ô "giữ"
                                                             │
                                                             ▼
                                            YOLO (chỉ chạy trên tile được giữ)
                                                             │
                                                             ▼
                                        Đổi tọa độ về ảnh gốc + NMS ──▶ kết quả
```

- **Nhãn huấn luyện:** ô (i, j) = 1 nếu có tâm của ít nhất một GT box nằm trong ô đó. GridNet học trực tiếp từ ảnh gốc + label YOLO, không cần dataset đã cắt tile.
- **Một mô hình cho mọi kích thước tile:** lưới 8×8 cố định. Khi đổi tile size, một tile được giữ nếu vùng nó phủ chạm vào bất kỳ ô "giữ" nào (`grid_mask_for_tiles` trong `dense_tiling_experiment.py`).
- **Hai chỉ số cần theo dõi khi train:**
  - `cell-recall`: tỉ lệ ô có vật thể được giữ lại. Cần rất cao, vì mỗi ô bị bỏ sót là một vật thể chắc chắn bị bỏ sót.
  - `keep-frac`: tỉ lệ ô được giữ. Càng thấp thì càng tiết kiệm.
- **Threshold** điều chỉnh cân bằng giữa hai chỉ số trên. Tham số `--pos-weight` cao hơn sẽ ưu tiên recall.

Chi tiết quy trình thí nghiệm và các quy tắc đo thời gian công bằng: xem [`RUNBOOK.md`](RUNBOOK.md).

---

## 3. Cấu trúc repo

### Phần chính của phase 2 (đọc trước)

| File / thư mục | Vai trò |
|---|---|
| `grid_proposal_net.py` | Định nghĩa, **train** và **benchmark** GridNet |
| `dense_tiling_experiment.py` | Thí nghiệm chính: so sánh **uniform với adaptive (GridNet)** trên nhiều tile size, xuất JSON và bảng break-even |
| `yolo_config/yolo26m-modified.yaml` | **Detector chính của phase 2**: YOLO26m + khối SE |
| `utils/custom_modules.py` | Module CBAM, SE (phải đăng ký với Ultralytics trước khi load model) |
| `tiling/tiling.py` | Cắt dataset thành tile uniform (offline) để train detector |
| `collect_report_metrics.py` | Gom metric nhiều seed thành bảng `report_metrics*.md` |
| `RUNBOOK.md` | Quy trình chạy thí nghiệm cho paper |

### Phần cũ / tham khảo

| File / thư mục | Ghi chú |
|---|---|
| `cnn_classifier/` | CNN chọn tile thế hệ 1 (hướng #2), kèm Grad-CAM (`small_cnn_heatmap.py`) |
| `tiling/adaptive_tiling_deprecated.py` | Hướng #1, **deprecated** |
| `evaluations/` | Script eval của pipeline CNN thế hệ 1 (`evaluation_with_cnn.py`, `evaluation_without_cnn.py`, …) |
| `bosch/`, `kayuan2024/` | Script train của phase 1 (export từ Colab, nhiều đường dẫn hardcode) |
| `train_seadronesee_*.py` | Train YOLOv8-CBAM trên SeaDroneSee (nhiều seed) |
| `yolo_config/YOLOv8_CBAM.yaml` | Detector YOLOv8 + CBAM |
| `utils/convert_*.py`, `utils/split_*.py` | Chuyển đổi và chia dataset |

### Data, weight và output

| Thư mục | Nội dung |
|---|---|
| `data/` | Dataset, bị gitignore: `data/kayuan/`, `data/SeaDroneSees/`, `data/SeaDroneSees_70_15_15/` |
| `weights/` | Weight GridNet, CNN thế hệ 1, YOLO. Nguồn gốc từng file ghi trong [`weights/README.md`](weights/README.md) |
| `outputs/<dataset>/` | Kết quả `dense_tiling_experiment.py` (`scaling*.json`), heatmap GridNet (`benchmark_results*/`), Grad-CAM, bảng `report_metrics*`. Xem [`outputs/README.md`](outputs/README.md) |
| `eval_logs/` | Log eval; `eval_logs/kayuan/{notile,tile,adaptive}/` cho Kayuan |
| `runs/` | Output train/val của Ultralytics, bị gitignore |

Các script mặc định ghi weight vào `weights/` và kết quả vào `outputs/` (chạy từ thư mục gốc repo). **Không để file kết quả ở thư mục gốc.**

---

## 4. Cài đặt

Yêu cầu: Python 3.10+, GPU NVIDIA (CUDA 12.x).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### Dataset

Tất cả dataset ở **định dạng YOLO** (`images/{train,val,test}`, `labels/{train,val,test}`, kèm file `data.yaml`).

| Dataset | Đối tượng | Số class | Nguồn tải | Đường dẫn trong repo |
|---|---|---|---|---|
| Kayuan | Đèn giao thông | 16 | `[TODO: link]` | `data/kayuan/` |
| SeaDroneSee | Người bơi, thuyền, phao… (ảnh drone) | 5 | `[TODO: link]` | `data/SeaDroneSees_70_15_15/` |
| VisDrone | `[TODO]` | `[TODO]` | `[TODO: link]` | `[TODO]` |
| Bosch (BSTLD) | Đèn giao thông | `[TODO]` | `[TODO: link]` | `[TODO]` |

---

## 5. Chạy thử

> ⚠️ Nhiều script vẫn hardcode đường dẫn tuyệt đối (`/home/vietpham/...`, `/home/anhld/...`). Kiểm tra và sửa đường dẫn trước khi chạy.

Ví dụ dưới đây dùng Kayuan. Với dataset khác, chỉ cần đổi đường dẫn.

**Bước 1: Train detector trên dataset đã cắt tile** (ảnh được cắt offline bằng `tiling/tiling.py`)

```
[TODO: lệnh train YOLO26m-modified trên dataset tiled]
```

**Bước 2: Train GridNet** (dùng ảnh gốc, không cần cắt tile)

```bash
python grid_proposal_net.py train \
    --images data/kayuan/train/images \
    --labels data/kayuan/train/labels \
    --grid-rows 8 --grid-cols 8 \
    --epochs 40 --pos-weight 10 \
    --out weights/gridnet_kayuan_8x8.pth
```

**Bước 3: Kiểm tra GridNet** (thời gian chạy, recall, và ảnh heatmap)

```bash
python grid_proposal_net.py benchmark \
    --model weights/gridnet_kayuan_8x8.pth \
    --images data/kayuan/val/images \
    --labels data/kayuan/val/labels \
    --threshold 0.10 \
    --benchmark-out outputs/kayuan/gridnet_val_thr010
```

**Bước 4: So sánh uniform với adaptive**

```bash
python dense_tiling_experiment.py \
    --images  data/kayuan/test/images \
    --labels  data/kayuan/test/labels \
    --model   <đường dẫn best.pt của detector ở bước 1> \
    --gridnet weights/gridnet_kayuan_8x8.pth \
    --tile-sizes 640 480 320 240 160 \
    --threshold 0.10 \
    --out outputs/kayuan/scaling/<tên_run>.json
```

Script sẽ in bảng **break-even** cho từng tile size: chênh lệch thời gian, mức tăng tốc, ΔmAP50 và Δrecall giữa adaptive và uniform.

---

## 6. Tiến độ & kết quả

### Trạng thái theo dataset

| Dataset | Có data | Detector (tiled) | GridNet đã train | Eval uniform vs adaptive |
|---|---|---|---|---|
| Kayuan | ⚠️ Mới có một phần (`label2`) | ⚠️ Mới có YOLOv8-CBAM (3 seed), chưa có YOLO26m-modified | ✅ `weights/gridnet_kayuan_8x8.pth` | ✅ 3 seed, thr 0.10 (cần chạy lại, xem mục 7) |
| SeaDroneSee | ✅ | ⚠️ Mới có YOLOv8-CBAM (3 seed), chưa có YOLO26m-modified | ✅ `weights/gridnet_seadronesee_8x8.pth` | ✅ 3 seed + sweep threshold |
| VisDrone | ❌ | ❌ | ❌ | ❌ |
| Bosch (BSTLD) | ❌ (chỉ có ở phase 1) | ❌ | ❌ | ❌ |

`[TODO: cập nhật trạng thái]`

### Kết quả

`[TODO: điền sau khi chạy lại với split mới và eval có xét class]`

| Dataset | Tile size | Tiles/ảnh (uniform) | Tiles/ảnh (adaptive) | Tăng tốc | mAP50 (uniform → adaptive) | Recall (uniform → adaptive) |
|---|---|---|---|---|---|---|
| | | | | | | |

---

## 7. Lưu ý quan trọng

1. **Hàm eval trong `dense_tiling_experiment.py` (`evaluate_accuracy`) chưa xét class.** Một prediction được ghép với GT chỉ dựa vào IoU, không kiểm tra class, và mAP đang được tính như chỉ có một class gộp chung. Trên Kayuan, lỗi này làm recall cao hơn khoảng 0.7pp. Phải sửa trước khi lấy số cho paper.
2. **Split Kayuan hiện tại bị rò rỉ dữ liệu.** Train/val/test được chia ngẫu nhiên theo **frame**, nên cả 68 chuỗi video đều xuất hiện ở cả 3 tập, và nhiều frame test chỉ cách frame train vài frame. Mọi metric tuyệt đối trên Kayuan đều bị cao hơn thực tế, dù so sánh uniform với adaptive vẫn công bằng. Cần chia lại **theo chuỗi video**. Split `SeaDroneSees_70_15_15` cũng nên kiểm tra lại theo cách này.
3. **Chỉ so sánh mAP giữa các phương pháp ở cùng tile size.** Tile size khác nhau thì độ khó của bài toán cũng khác.
4. **Phải đăng ký module custom trước khi load model có CBAM/SE:**
   ```python
   import ultralytics.nn.tasks as tasks
   from custom_modules import CBAM, SE   # utils/ phải có trong sys.path
   tasks.CBAM, tasks.SE = CBAM, SE
   ```
5. **Đường dẫn hardcode:** khoảng 13 file `.py` còn chứa đường dẫn tuyệt đối tới máy khác.

---

## 8. Liên hệ & trích dẫn

- Liên hệ: `[TODO]`
- Phase 1:

```bibtex
[TODO: BibTeX của AT-YOLO, DOI 10.1117/12.3120875]
```
