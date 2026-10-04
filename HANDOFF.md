# HANDOFF: tiếp tục công việc với Claude ở máy khác

> **Cách dùng:** mở session Claude mới trong thư mục repo và nói:
> *"Đọc HANDOFF.md rồi tiếp tục từ mục 'Việc tiếp theo'."*
>
> Viết ngày 2026-10-01, từ một session Claude Code trên server `anhld` (GPU 2× RTX 4090).

---

## 1. Bối cảnh project

- Repo này là **phase 2** của paper **AT-YOLO** (SPIE Proc. 14309, 143090V, DOI 10.1117/12.3120875): YOLO + tiling + attention (CBAM/SE).
- **Vấn đề:** tiling chậm vì YOLO phải chạy trên từng tile.
- **Mục tiêu phase 2:** chọn tile trước khi chạy YOLO để chạy nhanh hơn mà không mất độ chính xác.
- Các hướng đã thử:
  1. `tiling/adaptive_tiling_deprecated.py`: chọn số tile theo mật độ vật thể. **Deprecated**, không làm tiếp.
  2. `cnn_classifier/`: CNN phân loại từng tile. Chạy được nhưng **nghẽn ở batching**.
  3. **GridNet** (`grid_proposal_net.py`): **hướng chính**. Một lượt chạy trên cả ảnh cho ra lưới 8×8, chi phí cố định khoảng 10 ms/ảnh.
- GridNet **không làm tăng recall**, vì nó chỉ bỏ bớt tile. Đóng góp của nó là tốc độ ở cùng mức độ chính xác.

**Quyết định của chủ repo (user):**
- Detector chính của phase 2 là **YOLO26m-modified** (`yolo_config/yolo26m-modified.yaml`, YOLO26m + SE). Các kết quả hiện có lại dùng **YOLOv8-CBAM**.
- Dataset phase 2: **Kayuan** (đèn giao thông, 16 class; hiện chỉ có phần `label2`), **SeaDroneSee**, **VisDrone**, **Bosch (BSTLD)**. VisDrone và Bosch chưa có data trên server.
- README viết bằng **tiếng Việt**, người đọc chính là **thành viên mới**. Cài môi trường bằng **venv + requirements.txt**.
- Commit/PR: làm trên branch riêng, user review rồi mới commit, trừ khi user bảo commit.

## 2. Trạng thái repo

- Remote: `https://github.com/tuturuuu/TrafficLightDetectionVNDev.git`
- Branch: `ducanh-experiment` (upstream) → `docs/readme` → **`chore/cleanup-root`** (branch hiện tại)
- Commit mới nhất: `2bbdfd5c Reorganize repo root: weights/, outputs/, new README`. **Chưa push.** Khi tạo PR, chọn base là `ducanh-experiment`.
- **Chưa commit** (thuộc thí nghiệm Kayuan, định gom vào một commit riêng):
  - `cnn_classifier/small_cnn.py` (đã sửa)
  - `eval_logs/kayuan/`, `eval_logs/kayuan_gridnet_*.log`, `eval_logs/evaluation.txt`
  - `weights/gridnet_kayuan_8x8.pth`, `weights/tile_proposal_cnn_kayuan.pth`
  - `outputs/kayuan/kayuan_seed0_grid_test.json`

Cấu trúc sau khi dọn dẹp: xem `README.md` (mục 3), `weights/README.md`, `outputs/README.md`.

### Môi trường trên server cũ (để tham chiếu)
- Repo: `/home/anhld/AdaptiveTiling/TrafficLightDetectionVNDev`
- Python có ultralytics 8.4.32: `/opt/anaconda3/envs/traffic_light/bin/python` (python hệ thống không có cv2 hay ultralytics)
- Data (gitignored), tổng khoảng 56GB: `data/kayuan/` (trước là `label2_unzipped/`), `data/SeaDroneSees/`, `data/SeaDroneSees_70_15_15/`
- Các file `data*.yaml` **không còn dòng `path:`**. Ultralytics tự lấy thư mục chứa yaml làm gốc. Bản cũ được lưu ở `*.yaml.bak`.
- Script dọn dẹp và undo (ngoài repo): `/home/anhld/AdaptiveTiling/{cleanup_root.sh,undo_cleanup_root.sh,cleanup_root.log}`

## 3. Phát hiện quan trọng (chưa sửa)

### 3.1 Eval trong `dense_tiling_experiment.py` không xét class
`evaluate_accuracy` ghép prediction với GT **chỉ bằng IoU**, không kiểm tra class, và tính mAP như chỉ có một class gộp chung. Mình đã chạy lại trên Kayuan, tile 640, seed 0, GridNet thr 0.10, detector `runs/kayuan_yolov8_cbam_tiled_seed0/weights/best.pt`:

| | Cách cũ (không xét class) | Có xét class |
|---|---|---|
| Uniform | mAP50 0.9637, R 0.9942 | mAP50 0.9697, R 0.9874 |
| Adaptive | mAP50 0.9637, R 0.9935 | mAP50 0.9692, R 0.9868 |

Recall chỉ thấp hơn khoảng 0.7pp, và chênh lệch giữa adaptive và uniform không đổi. Dù vậy vẫn phải sửa trước khi lấy số cho paper. Riêng `evaluations/evaluation_with_cnn.py` có hàm `evaluate_predictions` (mAP50-95, dùng `iouv`) đáng tin hơn, nên lấy làm gốc khi gom eval. Hàm class-aware dùng để kiểm chứng có ở Phụ lục A.

### 3.2 Split Kayuan bị rò rỉ dữ liệu
- Train/val/test được chia ngẫu nhiên **theo frame**. Cả 68 chuỗi video (nhận ra qua tiền tố tên file, ví dụ `201809241107`, `20191106_082`, `201901220851`) đều có mặt ở cả 3 tập.
- Ví dụ: frame test `2019012208510010734` và frame train `2019012208510010727` là cùng một ngã tư, tọa độ đèn gần như trùng khớp.
- Hệ quả: mọi metric tuyệt đối trên Kayuan đều cao hơn thực tế (recall khoảng 0.99), còn so sánh adaptive với uniform vẫn công bằng.
- **Cần chia lại theo chuỗi video.** Split `SeaDroneSees_70_15_15` (từ `utils/split_seadrone_train_val_test.py`) cũng nên kiểm tra theo cách này.

### 3.3 Kết quả GridNet hiện có (Kayuan, thr 0.10, trung bình 3 seed, eval cũ không xét class)

| Tile | Tiles/ảnh | Giữ lại | Nhanh hơn (seed 0) | Recall uniform | Recall GridNet |
|---|---|---|---|---|---|
| 640 | 8 | 5.6 | 1.34× | 0.993 | 0.993 |
| 480 | 15 | 8.1 | 2.47× | 0.992 | 0.990 |
| 320 | 32 | 12.2 | 2.34× | 0.964 | 0.963 |
| 240 | 60 | 14.8 | 4.09× | 0.875 | 0.873 |
| 160 | 135 | 30.3 | 4.51× | 0.599 | 0.598 |

Nguồn: `eval_logs/kayuan/adaptive/gridnet/*.json`. Log không ghi detector đã dùng; nhiều khả năng là YOLOv8-CBAM.

SeaDroneSee (tile 640): GridNet giảm khoảng 58–63% số tile, chạy nhanh khoảng 1.7–2.3×, mAP gần như giữ nguyên (`outputs/seadronesee/scaling/`).

### 3.4 Các điểm khác
- `data/SeaDroneSees/data_tiled.yaml` **đã hỏng từ trước**: thư mục `*_tiled_uniform/` không tồn tại ở đó (chỉ có trong `SeaDroneSees_70_15_15`). File này lại là `DEFAULT_DATA` của `collect_report_metrics.py`.
- Chưa rõ nguồn gốc của các weight: `weights/gridnet_unknown_20260817.pth`, `weights/tile_proposal_cnn_model.pth`, `weights/tile_proposal_cnn_model_rootcopy.pth`. Hai file cuối cùng tên gốc nhưng nội dung khác nhau.
- Khoảng 13 file `.py` cũ còn hardcode đường dẫn `/home/vietpham/...`.

## 4. Việc tiếp theo: refactor theo pipeline ghép bằng config

User đồng ý hướng tiếp theo: giảm các file gần trùng nhau (yolo / yolo+tiling / yolo+tiling+cbam × cnn/gridnet × dataset) bằng **registry (factory) + builder + config ghép**. Thiết kế đề xuất, **user chưa chốt**:

```
Dataset ──▶ Tiler ──▶ TileSelector ──▶ Detector ──▶ Merger ──▶ Evaluator
            none       none             yolov8        nms        class-aware
            uniform    tile_cnn         yolov8_cbam              mAP50/50-95
                       gridnet          yolo26m_se               + timing
```

- Package `src/adaptile/` (tên tạm): `registry.py`, `tiling/`, `selectors/` (interface `select(image, tiles) -> bool mask`), `detectors/` (đăng ký CBAM/SE một lần), `merge/nms.py`, `eval/metrics.py`, `pipeline.py`.
- `configs/{dataset,detector,tiler,selector,experiment}/*.yaml`, ghép bằng **OmegaConf**. Hydra là phương án thay thế nếu cần sweep nhiều.
- Còn 4 script: `scripts/{train_detector,train_selector,evaluate,tile_dataset}.py`.

**Thứ tự làm**, không được làm thay đổi kết quả mà không biết:
1. Tách lõi dùng chung từ `dense_tiling_experiment.py` và `evaluations/evaluation_*.py`: tiling, chạy YOLO trên tile, NMS, đọc GT. Gom về **một hàm eval có xét class**.
2. Test kiểm tra kết quả: chạy pipeline mới và code cũ trên khoảng 50 ảnh; **prediction phải giống hệt**, metric chỉ khác ở phần class-aware.
3. Viết selector (none / tile_cnn / gridnet) và `scripts/evaluate.py`; kiểm tra lại số liệu Kayuan và SeaDroneSee.
4. Gom các script train vào `train_detector.py`.
5. Chuyển `bosch/`, `kayuan2024/`, `evaluations/` cũ vào `legacy/`.

Câu hỏi còn mở cho user: chọn OmegaConf hay Hydra, tên package, có làm trên branch mới từ `chore/cleanup-root` không.

Các việc khác trong backlog:
- Chia lại Kayuan theo chuỗi video, rồi train lại detector và GridNet.
- Train YOLO26m-modified, vì hiện chưa có run nào.
- Kiểm tra split SeaDroneSee.
- Điền các `[TODO]` trong README: link dataset, lệnh train YOLO26m, liên hệ, BibTeX.

---

## Phụ lục A: hàm eval có xét class đã dùng để kiểm chứng (mục 3.1)

Dùng cùng `_iou` của `dense_tiling_experiment.py`. GT có dạng `(x1, y1, x2, y2, cls)`, prediction có dạng `(x1, y1, x2, y2, score, cls)`.

```python
def ap101(records, n_gt):
    if not records or n_gt == 0:
        return 0.0, 0.0, 0.0
    records.sort(key=lambda r: r[0], reverse=True)
    tps = np.cumsum([r[1] for r in records]); fps = np.cumsum([not r[1] for r in records])
    prec = tps / (tps + fps); rec = tps / n_gt
    ap = np.mean([(prec[rec >= t].max() if (rec >= t).any() else 0.0)
                  for t in np.linspace(0, 1, 101)])
    return float(ap), float(prec[-1]), float(rec[-1])

def eval_class_aware(preds, gts, iou_thr=0.5):
    classes = sorted({g[4] for gg in gts for g in gg})
    aps, tp_all, n_pred, n_gt_all = [], 0, 0, 0
    for c in classes:
        recs, n_gt = [], 0
        for P, G in zip(preds, gts):
            g = [x for x in G if x[4] == c]; n_gt += len(g); m = [False] * len(g)
            for d in sorted([d for d in P if d[5] == c], key=lambda d: -d[4]):
                best, bi = 0.0, -1
                for j, gb in enumerate(g):
                    if not m[j] and (i := _iou(d, gb)) > best:
                        best, bi = i, j
                ok = best >= iou_thr and bi >= 0
                if ok: m[bi] = True
                recs.append((d[4], ok))
        ap, _, _ = ap101(recs, n_gt); aps.append(ap)
        tp_all += sum(r[1] for r in recs); n_pred += len(recs); n_gt_all += n_gt
    n_pred += sum(1 for P in preds for d in P if d[5] not in classes)  # FP of unseen classes
    return float(np.mean(aps)), tp_all / max(n_pred, 1), tp_all / max(n_gt_all, 1)
```
