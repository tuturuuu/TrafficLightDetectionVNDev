# HANDOFF: tiếp tục công việc với Claude ở máy khác

> **Cách dùng:** mở session Claude mới trong thư mục repo và nói:
> *"Đọc HANDOFF.md rồi tiếp tục từ mục 'Việc tiếp theo'."*
>
> Cập nhật lần cuối 2026-10-07, từ session Claude Code trên server `anhld` (2× RTX 4090).
>
> **Plan chi tiết nằm ở Notion:** [AdaptiveTiling phase 2: plan dataset → YOLO26 + CBAM → train → selector](https://app.notion.com/p/3f194a71622b819ab094e11c5abe58c0) (sửa lần cuối 06/10). File này chỉ tóm tắt; nếu hai nơi mâu thuẫn, **Notion là nguồn chính** cho plan, còn repo là nguồn chính cho trạng thái code.

---

## 1. Bối cảnh project

- Repo là **phase 2** của paper **AT-YOLO** (SPIE Proc. 14309, 143090V, DOI 10.1117/12.3120875): YOLO + tiling + attention (CBAM/SE).
- **Vấn đề:** tiling chậm vì YOLO phải chạy trên từng tile.
- **Mục tiêu phase 2:** chọn tile trước khi chạy YOLO để chạy nhanh hơn mà không mất độ chính xác.
- Các selector:
  - **GridNet** (`gridnet`): **hướng chính**. Một lượt chạy trên cả ảnh cho ra lưới 8×8, chi phí cố định mỗi ảnh.
  - CNN chọn từng tile (`tile_cnn`): thế hệ 1, chi phí tăng theo số tile.
  - Adaptive tiling theo mật độ vật thể: deprecated (`legacy/adaptive_tiling_deprecated.py`).
  - So sánh chi tiết các selector: `doc/selector.md`.
- GridNet **không làm tăng recall**, vì nó chỉ bỏ bớt tile. Đóng góp của nó là tốc độ ở cùng mức độ chính xác.

**Quyết định của user:**
- Detector chính của phase 2 là **YOLO26m**. Plan làm 4 biến thể để tách riêng đóng góp của tiling và attention: YOLO26 / +tiling / +tiling+CBAM / +tiling+SE.
- 4 dataset: Kayuan (đèn giao thông, 16 class), SeaDroneSee, VisDrone, Bosch (BSTLD).
- Tài liệu viết bằng **tiếng Việt**, người đọc chính là thành viên mới. Cài môi trường bằng venv + `requirements.txt`.
- Commit/PR: làm trên branch riêng; user review rồi mới commit, trừ khi user bảo commit.

## 2. Trạng thái repo

- Remote: `https://github.com/tuturuuu/TrafficLightDetectionVNDev.git`
- Branch hiện tại: **`refactor/pipeline`**, commit `6dcbd5c1 refactor folder structure` (04/10), **đã push** (khớp `origin/refactor/pipeline`). Theo Notion, PR đã mở với base `ducanh-experiment`.
  - Chuỗi branch: `ducanh-experiment` → `chore/cleanup-root` (`2bbdfd5c`, dọn thư mục gốc) → `refactor/pipeline`.
- **Chưa commit** (trên `refactor/pipeline`), là các sửa ngày 06/10 ghi trong Notion:
  - CNN selector resize giống lúc train: `resize: area` trong `configs/eval/*.yaml`. Chọn `bilinear_legacy` để tái lập số cũ.
  - Train GridNet/CNN: cố định `seed`, early stop theo val (`patience`) trong `configs/train/selector_*.yaml`, `src/adaptile/models/gridnet.py`, `legacy/cnn_classifier/small_cnn.py`.
  - `src/adaptile/selectors/tile_cnn.py`, `test/test_parity.py`, `test/test_training.py`.
  - Thư mục mới `doc/` (`selector.md`, `experiment_workflow.md`) và README có link tới đó.

### Cấu trúc sau refactor

```
src/adaptile/   package lõi: registry (TILERS/SELECTORS/DETECTORS/MERGERS), config (OmegaConf),
                tiling, selectors/{base,gridnet,tile_cnn}, detectors, merge, metrics,
                pipeline, training, models/{gridnet,tile_cnn,attention}
scripts/        train_detector.py, train_selector.py, evaluate.py, tile_dataset.py,
                benchmark_gridnet.py, collect_report_metrics.py, data/ (convert, split)
configs/        eval/<dataset>.yaml, train/{detector,selector}_<dataset>.yaml, models/*.yaml
test/           pytest (~1 phút; --full ~5 phút, cần GPU + data/kayuan + weights)
doc/            selector.md, experiment_workflow.md
legacy/         code cũ (dense_tiling_experiment.py, evaluations/, cnn_classifier/, bosch/, kayuan2024/, ...)
weights/        weight GridNet/CNN/YOLO pretrained, xem weights/README.md
outputs/        kết quả theo dataset, xem outputs/README.md
```

Mỗi thí nghiệm là **một file config**, ghi đè được bằng key có dấu chấm:

```bash
python scripts/evaluate.py configs/eval/kayuan.yaml tile_selector.type=[none,tile_cnn,gridnet] eval.max_images=50
python scripts/train_detector.py configs/train/detector_kayuan.yaml model.arch=yolo26m model.attention=cbam train.seeds=[0]
python scripts/train_selector.py configs/train/selector_kayuan.yaml out=weights/gridnet_kayuan_8x8_v2.pth
```

### Môi trường trên server cũ (để tham chiếu)
- Repo: `/home/anhld/AdaptiveTiling/TrafficLightDetectionVNDev`
- Python có ultralytics 8.4.32: `/opt/anaconda3/envs/traffic_light/bin/python` (python hệ thống không có cv2 hay ultralytics)
- Data (gitignored): `data/kayuan/` (chỉ phần `label2`), `data/SeaDroneSees/`, `data/SeaDroneSees_70_15_15/`, `data/visdrone/` (đã convert nhãn YOLO và tạo data.yaml ngày 2026-10-12). Trạng thái BSTLD xem cập nhật cuối file.
- Ổ `/home` còn trống khoảng **143 GB / 641 GB**. Mỗi dataset sau khi cắt tile tốn thêm khoảng gấp đôi.
- Các file `data*.yaml` không có dòng `path:`. Ultralytics tự lấy thư mục chứa yaml làm gốc.

## 3. Các phát hiện và trạng thái xử lý

| # | Vấn đề | Trạng thái |
|---|---|---|
| 3.1 | Eval cũ (`legacy/dense_tiling_experiment.py`) **không xét class** khi ghép prediction với GT, nên recall trên Kayuan cao hơn khoảng 0.7pp | ✅ **Đã sửa.** `src/adaptile/metrics.py` mặc định xét class (mAP50, mAP50-95, P, R). Metric `agnostic` cho ra số theo cách cũ để đối chiếu. **Không trộn** số cũ trong `outputs/*/scaling*.json` với số mới |
| 3.2 | **Split Kayuan bị rò rỉ dữ liệu**: chia theo frame, cả 68 chuỗi video đều có ở train/val/test (ví dụ test `2019012208510010734` và train `…10727` là cùng một ngã tư). Số tuyệt đối bị cao hơn thực tế (recall khoảng 0.99) | ✅ **Đã sửa 2026-10-12.** Tạo `data/kayuan_sequence` (14127/3036/3035), kiểm tra 73 chuỗi; SeaDroneSee cũng tạo split sạch `data/seadrone_sequence`. Không có nhóm nguồn giao nhau. Loại 51 ảnh Kayuan thiếu nhãn; config dùng split mới, GridNet dùng val riêng. **Phải train lại weight cũ** |
| 3.3 | YOLO26m + SE nối sai head, thiếu CBAM/P2 trong ARCHS | ✅ **Đã sửa 2026-10-12.** SE/CBAM Detect `[22,25,28,31]`, Concat cuối nối C2PSA; bổ sung YAML P2/SE/CBAM và ARCHS. Cả ba đã forward CPU với stride 4/8/16/32 |
| 3.4 | Eval thiếu latency, params/GFLOPs, peak GPU memory | ✅ **Đã bổ sung 2026-10-12.** Latency từng ảnh và p50/p95 select/detect/merge/total; params/GFLOPs theo model; peak allocated GPU memory sau warmup. Chưa benchmark GPU độc lập |
| 3.5 | `data/SeaDroneSees/data_tiled.yaml` trỏ tới `*_tiled_uniform/` không tồn tại, mà đây lại là `DEFAULT_DATA` của `scripts/collect_report_metrics.py` | ❌ Chưa sửa |
| 3.6 | Weight chưa rõ nguồn gốc: `gridnet_unknown_20260817.pth`, `gridnet_kayuan_8x8_rootcopy.pth`, `tile_proposal_cnn_model.pth`, `tile_proposal_cnn_model_rootcopy.pth` | ❓ Ghi trong `weights/README.md` |
| 3.7 | `scripts/data/convert_btsd_to_yolo.py` hardcode đường dẫn và chạy ngay khi được gọi (không có argparse) | ❌ Sửa trước khi dùng cho Bosch |

**Kết quả cũ đã mất:** log và JSON 3 seed GridNet trên Kayuan (trước ở `eval_logs/`) **đã bị xóa**. Nếu cần thì chạy lại bằng `scripts/evaluate.py configs/eval/kayuan.yaml` cho từng seed. Số tham khảo cũ (eval không xét class, split bị rò rỉ): ở tile 640, GridNet giữ 5.6/8 tile, nhanh hơn khoảng 1.3×, recall ≈ uniform (0.993). Ở tile 160, nhanh hơn khoảng 4.5×.

## 4. Việc tiếp theo

Làm theo plan Notion, theo thứ tự. Mỗi bước có một lượt eval chuẩn: tập test, conf 0.25, NMS 0.5, overlap 0.2, 3 seed (0/42/100), GPU không chạy job khác, chọn threshold trên **val**.

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 0. Chuẩn bị | Commit + PR refactor; thêm latency, params, GFLOPs, GPU mem vào `evaluate.py`; chia lại split Kayuan theo chuỗi video (3.2); Eval 0: chạy lại YOLOv8-CBAM trên split mới làm mốc | Commit + PR ✅; phần còn lại ❌ |
| 1. Dataset | Convert VisDrone sang YOLO (`scripts/data/convert_visdrone_to_yolo.py`; bỏ category 0 và 11, map 1–10 sang class 0–9, test-dev làm test); tải và convert Bosch (tách val theo chuỗi); config cho từng dataset; Eval 1: `scripts/data/dataset_stats.py` | ❌ |
| 2. YOLO26: 4 biến thể | Sửa head (3.3), thêm CBAM; Eval 2: params/GFLOPs/latency, train thử 1–5 epoch cả 4 biến thể | ❌ |
| 3. Train | Thứ tự ưu tiên: P0 GridNet mỗi dataset → P1 4 biến thể seed 0 trên Kayuan → P2 các dataset khác → P3 seed 42/100. Mục tiêu khoảng 3–4 giờ/run. Eval 3: bảng detector + bảng uniform vs GridNet theo tile size | ❌ |
| 4. Selector mới | Ứng viên: GridNet v2 (nhãn theo vùng box phủ, input 512, lưới 16×16), coarse detector (yolo26n), temporal (Kayuan là video), tile CNN đa tỉ lệ. Eval 4: Pareto recall theo speedup | ❌ |

**Quyết định đã chốt (2026-10-07): cắt bớt ma trận train**
- **3 seed (0/42/100) chỉ cho biến thể chính** (YOLO26 + tiling + CBAM hoặc SE, chọn sau Eval 2/3). Ba biến thể còn lại (YOLO26, YOLO26 + tiling, biến thể attention còn lại) **chỉ seed 0**, dùng làm ablation.
- Số run giảm từ 48 xuống **24** (4 dataset × (4 biến thể seed 0 + 2 seed thêm cho biến thể chính)), gồm 4 run không tiling và 20 run có tiling.
- Plan Notion **chưa cập nhật** quyết định này.

**Eval 2, đo thời gian train (đang chạy từ 2026-10-07):**
- `scripts/eval2_timing.sh` chạy 4 run 5 epoch trên Kayuan với `yolo26m-p2.yaml` (kiến trúc plan dùng cho 4 biến thể): {tiled, notile} × {pretrained `weights/yolo26m.pt`, scratch}. Mỗi GPU một hàng đợi; script **tự chờ GPU trống** (dưới 2 GB và dưới 10% util liên tục 3 phút) để thời gian đo không bị job khác làm sai. Lúc khởi chạy, cả 2 GPU đang bị user `henry` dùng.
- Log: `runs/eval2/logs/queue.log`, `runs/eval2/logs/<run>.log`. Tóm tắt: `python scripts/eval2_summary.py`. Kết quả nằm ở `runs/eval2/<run>_seed0/`.
- Mốc cũ: YOLOv8m-CBAM tiled trên Kayuan mất khoảng 29 phút/epoch (100 epoch là 56 giờ); bản notile 100 epoch là 5.8 giờ.
- Lưu ý: `yolo26m-p2` có 91.4 GFLOPs (`yolo26m` gốc là 75.4). Pretrained COCO chỉ nạp được 420/962 tensor vào bản P2 (backbone vào hết, neck/head phần lớn học lại). Trong khi đó `ARCHS` lại map `yolo26m+none` sang `yolo26m.yaml` **không có P2**, lệch với plan, cần sửa cùng lúc với 3.3.
- Val Kayuan đang bị rò rỉ (3.2), nên chỉ so **tương đối** pretrained với scratch.

**Việc làm ngay:**
1. Review và commit các thay đổi chưa commit (mục 2) trên `refactor/pipeline`.
2. Bước 0: chia lại split Kayuan theo chuỗi video; thêm latency, params, GFLOPs vào `evaluate.py`.
3. Sửa head `yolo26m-modified.yaml` (3.3), vì việc này chặn mọi run YOLO26.

**Câu hỏi còn mở cho user** (từ Notion, bước 3):
- Khởi tạo từ COCO pretrained (`yolo26m.pt`) hay train từ đầu.
- Số epoch và patience để đạt khoảng 3–4 giờ/run (run cũ dùng 100 epoch, patience 30).
- Một run dùng cả 2 GPU, hay mỗi GPU chạy một run.
- Kayuan có lấy thêm phần `label1` không.
- Skip connection ở P3/P4 lấy feature trước hay sau khối SE/CBAM.

## Cập nhật kiểm chứng 2026-10-12

Các mục 3.2/3.3/3.4 đã sửa trên branch `fix/handoff-3.2-3.4`, chưa commit. Chi tiết, lệnh tái tạo split và giới hạn đo nằm ở [doc/handoff_3_2_3_4.md](doc/handoff_3_2_3_4.md). Trạng thái cũ ở mục 4 là lịch sử; chưa train lại detector/selector hoặc chạy Eval 0 trên split mới.

## Chuẩn bị VisDrone / BSTLD (2026-10-12)

- VisDrone: đã tạo `data/visdrone/labels/{train,val,test}` và `data/visdrone/data.yaml`; test dùng test-dev. Số ảnh/nhãn: 6471/548/1610; bbox hợp lệ: 343204/38759/75102. Bỏ score=0, category 0/11 và 1 bbox train có chiều cao bằng 0. Script tái tạo: `python scripts/data/convert_visdrone_to_yolo.py`. 7 test converter đạt; kiểm tra toàn bộ nhãn đầu ra thành công.
- BSTLD: file `data/BSTLD/bstld.tar` tải qua dataset-tools thực tế là trang HTML lỗi Dropbox (Link Temporarily Disabled), không phải TAR. Nguồn thay thế chính thức: https://zenodo.org/records/12706046 (RGB train/test).
- Job BSTLD đã dừng theo yêu cầu user ngày 2026-10-12; các file tải dở `.part` được giữ lại, log `data/BSTLD/prepare.log`. Script `scripts/data/prepare_bstld.py` tải tiếp các `.part`, kiểm tra MD5 chính thức, rồi tự giải nén vào `data/BSTLD/raw/dataset_{train,test}_rgb`. Khi cả hai xong sẽ có `data/BSTLD/preparation_report.json`. Tải chưa hoàn tất, chưa giải nén; chỉ tiếp tục tải khi user yêu cầu.

## Hướng dẫn ablation Kayuan A1–A9 (2026-10-12)

- User đã chọn **YOLO26m-P2 cho toàn bộ bảng**, bao gồm baseline A1 không tiling.
- Hướng dẫn: `doc/kayuan_ablation_A1_A9.md`; 10 config train/eval cho A1/A2/A3/A6/A7 ở `configs/experiments/kayuan_p2/`. Bốn detector train riêng; CNN A3 dùng detector A2. Seed 0, scratch, split sequence mới; chọn threshold trên val trước test.
- A4 (activation averaging), A5/A8/A9 (LTS projection 1×1) chưa có implementation; không thay LTS bằng GridNet.
- Lệnh đã kiểm tra bằng dry-run và loader config; chưa khởi chạy training/eval GPU.


|Configuration| Tiling | Selection| Attn| 
|---|--|--|--|
|Baseline (yolo26m)|--|--|--|
|+Fixedtiling| Fixed|--|--|
|+Naivefilter| Fixed| CNN|--|
|+Param-freeavg| Grid| Avg|--|
|+LTS(learned)| Grid| LTS|--|
|+Fixed+CBAM| Fixed|--| CBAM |
|+Fixed+SE| Fixed|--| SE|
|+LTS+CBAM| Grid| LTS| CBAM 
|+LTS+SE(LAT-YOLO)| Grid| LTS| SE|