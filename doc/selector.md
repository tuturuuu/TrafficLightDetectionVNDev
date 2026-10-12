# Tile selector: GridNet và CNN selector

Cả hai đều là **tile selector**: đứng giữa bước cắt tile và YOLO, quyết định tile nào cần đưa qua YOLO. Mục tiêu chung là bỏ bớt tile trống (trời, đường, mặt biển) để YOLO chạy ít lần hơn.

Vì chỉ bỏ tile, selector **không thể làm tăng** recall. Tốt nhất là giữ nguyên độ chính xác như uniform tiling mà chạy nhanh hơn. Khác biệt cốt lõi giữa hai cái là **nhìn ảnh theo đơn vị nào**.

Trong pipeline, selector được chọn bằng `tile_selector.type` trong `configs/eval/*.yaml` (`none` | `tile_cnn` | `gridnet`), và train bằng `scripts/train_selector.py` với `configs/train/selector_*.yaml`.

## 1. CNN selector (`tile_cnn`): thế hệ 1, chấm điểm từng tile

```
Ảnh 1920×1080 ──cắt──▶ N tile ──▶ resize mỗi tile về 160×160 ──▶ CNN ──▶ N điểm (0..1)
                                                                          │ > threshold (0.5)
                                                                          ▼
                                                         giữ tile có điểm cao ──▶ YOLO
```

- **Model** ([`src/adaptile/models/tile_cnn.py`](../src/adaptile/models/tile_cnn.py)), 74,641 tham số:
  - 6 lớp conv (16 → 32 → 64 kênh) cộng 2 lần max-pool.
  - Sau đó là global average pool và MLP 64 → 32 → 1, cho ra **một con số cho mỗi tile**: xác suất tile đó có vật thể.
- **Tiền xử lý** ([`src/adaptile/selectors/tile_cnn.py`](../src/adaptile/selectors/tile_cnn.py)): crop tile, pad (114) cho vuông, BGR → RGB, `cv2.resize` INTER_AREA về 160×160, chia 255, **giống hệt lúc train**. Mọi tile của một ảnh nằm chung một batch.
  - Code cũ (`legacy/evaluations/evaluation_with_cnn.py`) resize bilinear không antialias trên GPU, lệch với lúc train: đo trên 1623 tile val Kayuan, tile-recall ở threshold 0.5 là 0.927 so với 0.947 khi resize đúng, cùng tỉ lệ tile giữ lại. Đặt `resize: bilinear_legacy` trong config để tái lập số cũ.
- **Nhãn khi train:** tile = 1 nếu file label của tile có ít nhất một box.
  - Train trên dataset **đã cắt tile sẵn** (`train_tiled_uniform/`), tức là chỉ một tile size (640).
  - Script train: `legacy/cnn_classifier/small_cnn.py`, gọi qua `scripts/train_selector.py` với `type: tile_cnn`. Early stopping, giảm LR và chọn checkpoint đều theo loss trên tập val (`val_images`); lưu epoch có val loss thấp nhất.
- **Chi phí tăng theo số tile.** Ảnh có N tile thì phải crop, resize và chạy CNN N lần.
  - Ở tile 640 (8 tile/ảnh) thì không đáng kể.
  - Ở tile 160 (135 tile/ảnh) thì phần chi phí này ăn mất phần lớn thời gian tiết kiệm được ("nghẽn ở batching").
- **Không có safety floor.** Nếu CNN cho mọi tile điểm thấp, ảnh đó không được chạy YOLO chút nào (giống code cũ `legacy/evaluations/evaluation_with_cnn.py`).

## 2. GridNet (`gridnet`): hướng chính, một lượt trên cả ảnh

```
Ảnh 1920×1080 ──resize 256×256──▶ GridNet (1 lượt) ──▶ lưới 8×8 xác suất
                                                          │ > threshold (vd 0.10), giữ ít nhất 1 ô
                                                          ▼
                                                  mask 8×8 ô "giữ"
Ảnh ──cắt N tile (uniform)──▶ giữ tile nào phủ lên ít nhất 1 ô "giữ" ──▶ YOLO
```

- **Model** ([`src/adaptile/models/gridnet.py`](../src/adaptile/models/gridnet.py)), 48,945 tham số (lưới 8×8):
  - 4 khối conv stride 2 (16 → 32 → 48 → 64 kênh) đưa ảnh từ 256×256 xuống feature map 16×16.
  - `AdaptiveAvgPool` về 8×8, rồi 2 lớp conv 1×1, cho ra **64 điểm, mỗi điểm ứng với một ô** của lưới 8×8 phủ cả ảnh.
- **Nhãn khi train:** ô (i, j) = 1 nếu **tâm** của ít nhất một GT box rơi vào ô đó.
  - Train trực tiếp trên ảnh gốc cộng label YOLO, không cần cắt tile. 10% ảnh train (cố định, seed 0) dùng làm val.
  - `pos_weight` (vd 10) phạt nặng việc bỏ sót ô có vật thể, nên model ưu tiên recall.
- **Một model dùng cho mọi tile size** (`grid_mask_for_tiles` trong [`src/adaptile/tiling.py`](../src/adaptile/tiling.py)):
  - Lưới 8×8 cố định trên ảnh, nên với Kayuan 1920×1080 mỗi ô rộng 240×135 px.
  - Khi cắt tile cỡ bất kỳ, một tile được giữ nếu vùng pixel của nó **chồng lên bất kỳ ô "giữ" nào** (lấy OR qua các ô mà tile phủ lên).
- **Chi phí cố định khoảng 10–14 ms/ảnh**, dù ảnh có 8 hay 135 tile. Vì vậy tile càng nhỏ thì GridNet càng có lợi.
- **Luôn giữ ít nhất 1 ô** (`min_keep=1`), nên ảnh nào cũng được chạy YOLO ít nhất một tile.

## 3. So sánh

| | CNN selector | GridNet |
|---|---|---|
| Đơn vị quyết định | Từng tile | Ô của lưới 8×8 cố định |
| Số lượt chạy mỗi ảnh | N (bằng số tile) | 1 |
| Chi phí khi tile nhỏ đi | Tăng theo N | Không đổi |
| Độ phân giải đầu vào | 160×160 mỗi tile, nhìn chi tiết | 256×256 cho cả ảnh, nhìn thô |
| Dữ liệu train | Dataset đã cắt tile, một tile size | Ảnh gốc, không phụ thuộc tile size |
| Đổi tile size | Lệch phân phối so với lúc train | Dùng được ngay |
| Không chọn được tile nào | Bỏ qua ảnh | Vẫn giữ ít nhất 1 ô |

**Số liệu chạy thử** (Kayuan, detector YOLOv8-CBAM seed 0, GridNet thr 0.10, CNN thr 0.5; 20–50 ảnh đầu tập test, chỉ để tham khảo; CNN dùng cách resize cũ):

| Tile | Selector | Tile giữ lại / ảnh | Nhanh hơn uniform | Δrecall |
|---|---|---|---|---|
| 640 | CNN | ~3.1 / 8 | ~1.8× | −2 đến −3pp |
| 640 | GridNet | ~6 / 8 | ~1.1–1.2× | ~0 |
| 320 | CNN | ~2 / 32 | ~11× | **−12 đến −17pp** |
| 320 | GridNet | ~13 / 32 | ~2.4× | ~0 |

- Ở tile 320, CNN bỏ gần hết tile và mất nhiều recall. Nhiều khả năng là do CNN được train trên tile 640, nên tile 320 nằm ngoài phân phối lúc train.
- Kết quả cũ trên toàn bộ tập test (2032 ảnh, tile 160, eval không xét class): GridNet giữ khoảng 30/135 tile, nhanh hơn khoảng 4.5×, recall chỉ giảm 0.09pp.

## 4. Điểm yếu cần biết của GridNet

- **Lưới thô.** Một ô 240×135 px chứa vật thể thì mọi tile chạm vào ô đó đều phải giữ. Tile càng nhỏ, tỉ lệ tile bị "kéo theo" càng cao, nên vẫn có thể giảm thêm số tile.
- **Nhãn chỉ dựa vào tâm box.** Box nằm vắt qua ranh giới hai ô chỉ đánh dấu ô chứa tâm. Phần box nằm ở ô bên cạnh có thể rơi vào tile bị bỏ. Hiện chưa thấy vấn đề lớn vì recall gần như không đổi.
- **Threshold quyết định cân bằng giữa recall và tốc độ.** Threshold thấp thì giữ nhiều tile, an toàn nhưng chậm hơn. Khi train cần theo dõi hai chỉ số:
  - `cell-recall`: tỉ lệ ô có vật thể được giữ lại, phải gần 1.
  - `keep-frac`: tỉ lệ ô được giữ, càng thấp càng tiết kiệm.

## Tóm lại

CNN selector hỏi "tile này có vật thể không?" N lần cho mỗi ảnh. GridNet hỏi "vùng nào của ảnh có vật thể?" đúng một lần, rồi suy ra tile nào cần giữ cho bất kỳ tile size nào. Vì vậy GridNet là hướng chính của phase 2.
