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
| 1 | Adaptive tiling theo mật độ vật thể (NBA) | `legacy/adaptive_tiling_deprecated.py` | ❌ **Deprecated** | Không nghiên cứu tiếp |
| 2 | CNN phân loại từng tile (sau khi cắt) | `legacy/cnn_classifier/`, selector `tile_cnn` | ⚠️ Thay thế bởi #3 | Chọn tile ổn, nhưng nghẽn ở khâu batching vì phải chạy CNN trên từng tile, nên chi phí tăng theo số tile |
| 3 | **GridNet**: một lượt chạy trên cả ảnh | `src/adaptile/models/gridnet.py` | ✅ **Hướng chính** | Bản cải tiến của #2, không bị nghẽn batching, chi phí cố định mỗi ảnh |

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

```
configs/
  train/      detector_<dataset>.yaml, selector_<dataset>.yaml   # mỗi lần train: một file
  eval/       <dataset>.yaml                                     # mỗi thí nghiệm: một file
  models/     YOLOv8_CBAM.yaml, yolo26m-modified.yaml            # kiến trúc YOLO + attention
scripts/
  train_detector.py   train_selector.py   evaluate.py            # 3 lệnh chính
  benchmark_gridnet.py  collect_report_metrics.py  tile_dataset.py
  data/       convert_*.py, split_*.py                           # chuẩn bị dataset
src/adaptile/   code dùng chung: tiling, selector, detector, NMS, metric, pipeline, train
test/           pytest
legacy/         code cũ đã được thay thế (xem legacy/README.md)
weights/  outputs/      weight đã train, kết quả thí nghiệm
data/  runs/            dataset, output Ultralytics (gitignore)
```

| File / thư mục | Vai trò |
|---|---|
| `scripts/train_detector.py` | **Train detector** từ một config: chọn dataset, tiling (uniform/none), kiến trúc (yolov8m/yolo26m), attention (none/cbam/se), seed. Tự cắt tile nếu chưa có |
| `scripts/train_selector.py` | **Train tile selector** (GridNet hoặc CNN từng tile) từ một config |
| `scripts/evaluate.py` | **Thí nghiệm chính**: so sánh uniform với các selector trên nhiều tile size, xuất JSON và bảng so sánh |
| `scripts/benchmark_gridnet.py` | Benchmark GridNet: thời gian mỗi ảnh, cell-recall, ảnh heatmap |
| `scripts/collect_report_metrics.py` | Gom metric nhiều seed (từ `runs/`) thành bảng `report_metrics*.md` |
| `scripts/tile_dataset.py` | Cắt dataset thành tile bằng tay (vd tập test). `train_detector.py` đã tự cắt train/val |
| `scripts/data/` | Chuyển COCO/BTSD sang YOLO, chia train/val/test SeaDroneSee |
| `configs/models/yolo26m-modified.yaml` | **Detector chính của phase 2**: YOLO26m + khối SE. `YOLOv8_CBAM.yaml`: YOLOv8m + CBAM |
| `src/adaptile/` | Pipeline Tiler → TileSelector (`none` / `gridnet` / `tile_cnn`) → Detector → NMS → eval có xét class, và phần train. Thêm thành phần mới bằng `@SELECTORS.register("tên")` (xem `registry.py`). CBAM/SE ở `models/attention.py`, GridNet ở `models/gridnet.py` |
| `test/` | Test đơn vị, test so khớp pipeline mới với code cũ, test train (xem mục 4) |
| `legacy/` | Code cũ: `dense_tiling_experiment.py`, `evaluations/`, `cnn_classifier/` (CNN thế hệ 1, Grad-CAM; `small_cnn.py` vẫn được `train_selector.py` gọi khi `type: tile_cnn`), script train phase 1, … Xem [`legacy/README.md`](legacy/README.md) |
| `weights/` | Weight GridNet, CNN thế hệ 1, YOLO. Nguồn gốc từng file ghi trong [`weights/README.md`](weights/README.md) |
| `outputs/<dataset>/` | Kết quả `scripts/evaluate.py` và kết quả cũ (`scaling*.json`), heatmap GridNet, Grad-CAM, bảng `report_metrics*`. Xem [`outputs/README.md`](outputs/README.md) |
| `RUNBOOK.md` | Quy trình chạy thí nghiệm cho paper |

Các script mặc định ghi weight vào `weights/` và kết quả vào `outputs/` (chạy từ thư mục gốc repo). **Không để file kết quả ở thư mục gốc.**

---

## 4. Cài đặt

Yêu cầu: Python 3.10+, GPU NVIDIA (CUDA 12.x).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # cài package adaptile (src/adaptile) ở chế độ editable
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Chạy test (`test/test_units.py` không cần GPU; `test/test_parity.py` so pipeline mới với `legacy/dense_tiling_experiment.py` trên ảnh Kayuan thật và tự skip nếu thiếu GPU/data/weights):

```bash
pytest test/                 # khoảng 1 phút
pytest test/ --full          # thêm so khớp trên toàn bộ tập test Kayuan và train thử 1 epoch, khoảng 5 phút
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

Mọi bước đều chạy từ thư mục gốc repo, mỗi bước một file config. Sửa file, hoặc ghi đè nhanh từ dòng lệnh bằng `key=value`. Thêm `--dry-run` vào lệnh train để chỉ in tham số. Ví dụ dưới đây dùng Kayuan; với SeaDroneSee, dùng file `*_seadronesee.yaml` tương ứng.

**Bước 1: Train detector** ([`configs/train/detector_kayuan.yaml`](configs/train/detector_kayuan.yaml))

```yaml
dataset:  {data_yaml: data/kayuan/data.yaml}
tiling:   {type: uniform, tile_size: 640, overlap: 0.2}   # uniform | none
model:    {arch: yolov8m, attention: cbam}                 # yolov8m | yolo26m;  none | cbam | se
train:    {seeds: [0, 42, 100], epochs: 100, batch: 16, device: 0}
output:   {project: runs, name: kayuan_yolov8_cbam_tiled_v2}
```

```bash
python scripts/train_detector.py configs/train/detector_kayuan.yaml
python scripts/train_detector.py configs/train/detector_kayuan.yaml model.arch=yolo26m model.attention=se output.name=kayuan_yolo26m_se_tiled
```

Với `tiling.type: uniform`, script tự cắt train/val thành tile (nếu `<split>_tiled_uniform/` chưa có) rồi train trên tile. Weight của từng seed nằm ở `runs/<output.name>_seed<seed>/weights/best.pt`.

**Bước 2: Train GridNet** ([`configs/train/selector_kayuan.yaml`](configs/train/selector_kayuan.yaml); GridNet dùng ảnh gốc, không cần cắt tile)

```bash
python scripts/train_selector.py configs/train/selector_kayuan.yaml
python scripts/train_selector.py configs/train/selector_kayuan.yaml type=tile_cnn out=weights/tile_cnn_kayuan_v2.pth
```

**Bước 3: Kiểm tra GridNet** (thời gian chạy, recall, và ảnh heatmap)

```bash
python scripts/benchmark_gridnet.py benchmark \
    --model weights/gridnet_kayuan_8x8.pth \
    --images data/kayuan/val/images \
    --labels data/kayuan/val/labels \
    --threshold 0.10 \
    --benchmark-out outputs/kayuan/gridnet_val_thr010
```

**Bước 4: So sánh uniform với adaptive**

Mỗi thí nghiệm là **một file config** trong `configs/` (vd [`configs/eval/kayuan.yaml`](configs/eval/kayuan.yaml)), chứa đủ thông tin của từng phần, có chú thích các lựa chọn:

| Mục | Nội dung chính |
|---|---|
| `dataset` | `images`, `labels` |
| `tiling` | `type: uniform \| none`, `tile_size` (một số hoặc danh sách để quét), `overlap` |
| `tile_selector` | `type: none \| tile_cnn \| gridnet` (một hoặc danh sách để so sánh), kèm mục `gridnet:` / `tile_cnn:` chứa `weights`, `threshold` |
| `detector` | `type: yolo`, `weights` (`best.pt` của YOLOv8-CBAM, YOLO26m-SE, …), `conf`, `imgsz` |
| `merger` | `type: nms \| none`, `iou_thr`, `max_det` |
| `eval` | `metrics` (`map50`, `map50_95`, `precision`, `recall`, `agnostic`, `fps`), `max_images`, `out` |

Thí nghiệm mới: copy một file rồi sửa. Muốn đổi nhanh thì ghi đè từ dòng lệnh bằng key có dấu chấm:

```bash
python scripts/evaluate.py configs/eval/kayuan.yaml
python scripts/evaluate.py configs/eval/kayuan.yaml tile_selector.gridnet.threshold=0.05 tiling.tile_size=[640,320]
python scripts/evaluate.py configs/eval/kayuan.yaml tile_selector.type=[none,tile_cnn,gridnet] eval.max_images=50 eval.out=/tmp/thu.json
```

Script in bảng so với uniform cho từng tile size (mức tăng tốc, ΔmAP50, Δrecall), rồi lưu JSON vào `eval.out` cùng file `.yaml` ghi lại config đã chạy. `mAP50`, `precision`, `recall` trong JSON **có xét class**; số tính theo cách cũ của `legacy/dense_tiling_experiment.py` nằm trong `agnostic`.

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

1. **Số cũ không xét class.** Hàm eval của `legacy/dense_tiling_experiment.py` (`evaluate_accuracy`) ghép prediction với GT chỉ bằng IoU, không kiểm tra class, nên mọi kết quả cũ trong `outputs/*/scaling*.json` đều tính theo cách này. Trên Kayuan, cách tính cũ làm recall cao hơn khoảng 0.7pp. `scripts/evaluate.py` đã **xét class** và có thêm mAP50-95; dùng số từ script này cho paper. Muốn đặt cạnh số cũ thì thêm `agnostic` vào `eval.metrics`.
2. **Split Kayuan hiện tại bị rò rỉ dữ liệu.** Train/val/test được chia ngẫu nhiên theo **frame**, nên cả 68 chuỗi video đều xuất hiện ở cả 3 tập, và nhiều frame test chỉ cách frame train vài frame. Mọi metric tuyệt đối trên Kayuan đều bị cao hơn thực tế, dù so sánh uniform với adaptive vẫn công bằng. Cần chia lại **theo chuỗi video**. Split `SeaDroneSees_70_15_15` cũng nên kiểm tra lại theo cách này.
3. **Chỉ so sánh mAP giữa các phương pháp ở cùng tile size.** Tile size khác nhau thì độ khó của bài toán cũng khác.
4. **Load model có CBAM/SE bằng code riêng** (notebook, script ngoài pipeline) thì phải đăng ký module trước:
   ```python
   from adaptile.detectors import register_custom_modules
   register_custom_modules()          # trước YOLO(...)
   ```
5. **Đường dẫn hardcode:** code ngoài `legacy/` không còn đường dẫn tuyệt đối, trừ `scripts/data/convert_btsd_to_yolo.py`. Các file trong `legacy/` (script phase 1, Grad-CAM, …) vẫn còn đường dẫn tới máy khác.

---

## 8. Liên hệ & trích dẫn

- Liên hệ: `[TODO]`
- Phase 1:

```bibtex
[TODO: BibTeX của AT-YOLO, DOI 10.1117/12.3120875]
```
