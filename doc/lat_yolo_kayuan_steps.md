# Chạy LAT-YOLO trên Kayuan, từng bước

Luồng chính: **YOLO26m P2–P5 trên ảnh nguyên -> YOLO26m P2–P5 trên tile ->
LTS joint -> chọn threshold trên validation -> test ba seed -> tổng hợp metrics**.

Ví dụ dùng attention **SE**, tức biến thể LAT-YOLO trong spec. Để chạy CBAM hoặc
không attention, đổi hai biến ở bước 1 và chạy lại với các tên output tương ứng.
Các lệnh training dưới đây là để bạn thực hiện; trong lần hướng dẫn này chỉ kiểm
tra bằng dry-run, chưa chạy training GPU.

## 1. Môi trường, dataset và biến dùng chung

Chạy các bước trong cùng một Bash shell:

```bash
cd /home/anhld/AdaptiveTiling/TrafficLightDetectionVNDev
conda activate traffic_light
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

DATA_ROOT="data/kayuan_sequence"
RUN_ROOT="runs/lat_yolo/kayuan"
ATTN=se
FEATURE_STAGE=5
PW=1
LAMBDA=1
RECIPE="${ATTN}_pw${PW}_lam${LAMBDA}"
RESULT_ROOT="outputs/lat_yolo/kayuan/${RECIPE}"
```

GPU vật lý 1 được ánh xạ thành GPU 0 trong process. Vì vậy `train_detector.py`
dùng `train.device=0`; LTS và evaluator dùng `cuda:0`.

| Biến thể | ATTN | FEATURE_STAGE |
|---|---|---|
| YOLO26m P2–P5 | none | 4 |
| YOLO26m P2–P5 + CBAM | cbam | 5 |
| YOLO26m P2–P5 + SE | se | 5 |

Dataset hiện có:

```text
data/kayuan_sequence/
  data.yaml                 # 16 class
  split_manifest.json
  train/images/  train/labels/
  val/images/    val/labels/
  test/images/   test/labels/
```

Nếu chưa có dataset (máy mới), tải Kayuan `label2` từ Google Drive và tạo lại
split này bằng một lệnh. Cần Drive API key trong `.env` (`GDRIVE_API_KEY=...`),
xem README mục Dataset:

```bash
scripts/data/fetch_kayuan_gdrive.sh
```

P1.1 đã được bạn giải quyết. Các bước dưới dùng split này. Checkpoint YOLO26 đang
có ở `runs/eval2/...` là thí nghiệm 5 epoch trên `data/kayuan`; quy trình mới tạo
checkpoint riêng trên `kayuan_sequence`.

Package cần `adaptile`, Ultralytics và pycocotools. Nếu chưa cài package editable
và dependency COCO trong environment hiện tại:

```bash
python -m pip install -e '.[research]'
```

Giữ Ultralytics 8.4.32 đã được kiểm tra với implementation. `PW=1`, `LAMBDA=1`
là điểm khởi đầu để thực nghiệm. Pos-weight tối ưu của GridNet không tự động trở
thành pos-weight tối ưu của LTS, vì target và feature khác nhau.

## 2. Train detector trên ảnh nguyên, ba seed

Kiểm tra lệnh trước:

```bash
python scripts/train_detector.py configs/train/detector_kayuan.yaml \
  dataset.data_yaml="$DATA_ROOT/data.yaml" \
  model.arch=yolo26m model.attention="$ATTN" model.pretrained=null \
  tiling.type=none train.device=0 'train.seeds=[0,42,100]' \
  output.project="$RUN_ROOT" output.name="yolo26m_p2_notile_${ATTN}" \
  --dry-run
```

Sau khi kiểm tra, chạy lại đúng lệnh trên và bỏ `--dry-run`. Config hiện đặt
100 epoch, patience 30. Các lệnh detector trong tài liệu khởi tạo từ đầu; nếu
dùng pretrained, ghi rõ và áp dụng cùng chính sách cho các detector so sánh.

Weight được lưu tại:

```text
runs/lat_yolo/kayuan/yolo26m_p2_notile_se_seed0/weights/best.pt
runs/lat_yolo/kayuan/yolo26m_p2_notile_se_seed42/weights/best.pt
runs/lat_yolo/kayuan/yolo26m_p2_notile_se_seed100/weights/best.pt
```

## 3. Train detector trên uniform tile, ba seed

```bash
python scripts/train_detector.py configs/train/detector_kayuan.yaml \
  dataset.data_yaml="$DATA_ROOT/data.yaml" \
  model.arch=yolo26m model.attention="$ATTN" model.pretrained=null \
  tiling.type=uniform tiling.tile_size=640 tiling.overlap=0.2 \
  tiling.suffix=_lat640_ov20_v1 \
  train.device=0 'train.seeds=[0,42,100]' \
  output.project="$RUN_ROOT" output.name="yolo26m_p2_tiled_${ATTN}" \
  --dry-run
```

Bỏ `--dry-run` để thực hiện. Script tự cắt train/val nếu chưa có. Không cần chạy
`dataset_tiling.py` riêng. Tile được tạo vào:

```text
data/kayuan_sequence/train_lat640_ov20_v1/{images,labels}/
data/kayuan_sequence/val_lat640_ov20_v1/{images,labels}/
```

Suffix riêng giúp giữ lại dataset tiled trước đó. Nếu đổi tile size, overlap hoặc
visibility, dùng suffix khác. Weight tương ứng là:

```text
runs/lat_yolo/kayuan/yolo26m_p2_tiled_se_seed0/weights/best.pt
runs/lat_yolo/kayuan/yolo26m_p2_tiled_se_seed42/weights/best.pt
runs/lat_yolo/kayuan/yolo26m_p2_tiled_se_seed100/weights/best.pt
```

## 4. Frozen pilot trên seed 0, bước kiểm tra tùy chọn

Đây là kiểm tra khả năng học relevance từ backbone đã train. Chỉ head LTS học,
detector được giữ cố định. Dùng một run riêng với 5 epoch:

```bash
python scripts/train_lts.py configs/train/lts_kayuan.yaml \
  detector.weights="$RUN_ROOT/yolo26m_p2_tiled_${ATTN}_seed0/weights/best.pt" \
  features.feature_stage="$FEATURE_STAGE" \
  training_mode=frozen_head_pilot train.seed=0 train.device=cuda:0 \
  train.epochs=5 train.pos_weight="$PW" train.lambda_rel="$LAMBDA" \
  out="$RUN_ROOT/lts_frozen_${RECIPE}_seed0"
```

Theo dõi loss, cell recall, positive fraction và **kept_tile_fraction**. Pilot
frozen có nhãn riêng trong checkpoint. Bước joint tiếp theo khởi tạo head mới;
nó không tự nạp head của pilot.

## 5. Train joint trên seed 0

LTS joint tối ưu loss detection của YOLO cùng relevance BCE trên backbone chung:
`L_det + lambda_rel * L_rel`. Fine loss dùng tile được lấy mẫu từ cùng ảnh gốc
đang cung cấp supervision cho coarse branch.

```bash
python scripts/train_lts.py configs/train/lts_kayuan.yaml \
  detector.weights="$RUN_ROOT/yolo26m_p2_tiled_${ATTN}_seed0/weights/best.pt" \
  features.feature_stage="$FEATURE_STAGE" \
  training_mode=joint train.seed=0 train.device=cuda:0 \
  train.pos_weight="$PW" train.lambda_rel="$LAMBDA" \
  out="$RUN_ROOT/lts_joint_${RECIPE}_seed0"
```

Config đặt tối đa 40 epoch, batch 4 và patience 10. `train.threshold=0.5` chỉ dùng
cho diagnostic khi train; hard selection không quyết định tile cho detection loss.

Run có:

```text
lts_joint_se_pw1_lam1_seed0/
  config.json
  history.json
  resume.pt
  best.json
  best_epochNNNN/
    detector.pt
    lts.pt
    metadata.json
```

`best.json` chỉ tới epoch được chọn theo validation loss. Joint thay đổi trọng số
detector, nên evaluation phải dùng `detector.pt` xuất cùng `lts.pt`.

Khai báo helper để đọc đúng đường dẫn bằng JSON parser:

```bash
pair_for_seed() {
  python - "$RUN_ROOT/lts_joint_${RECIPE}_seed$1" <<'PY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1])
print(run / json.loads((run / "best.json").read_text())["directory"])
PY
}
PAIR=$(pair_for_seed 0)
```

Nếu cần resume, chạy lại đúng lệnh training và thêm
`--resume "$RUN_ROOT/lts_joint_${RECIPE}_seed0/resume.pt"`.
Giữ nguyên config, source checkpoint và source code của run khi resume.

## 6. Chọn threshold trên validation

Đánh giá uniform, param-free và LTS với cùng detector joint seed 0:

```bash
for THR in 0.3 0.5 0.7 0.9; do
  python scripts/benchmark_lts.py configs/eval/lts_kayuan.yaml \
    detector.device=cuda:0 detector.weights="$PAIR/detector.pt" \
    tile_selector.lts.weights="$PAIR/lts.pt" \
    tile_selector.lts.feature_stage="$FEATURE_STAGE" \
    tile_selector.param_free_avg.feature_stage="$FEATURE_STAGE" \
    tile_selector.lts.threshold="$THR" \
    tile_selector.param_free_avg.threshold="$THR" \
    eval.seed=0 eval.profile_stages=false eval.visualize=0 \
    eval.experiment_id="${RECIPE}_val_thr${THR}" \
    eval.out="$RESULT_ROOT/val_seed0_thr${THR}.json" || break
done
```

Chọn threshold theo mAP, recall, số tile giữ lại và FPS trên validation. Xác định
mức suy giảm chất lượng chấp nhận được trước khi chọn. Hai selector có phân bố
score khác nhau, nên có thể chọn threshold khác nhau; so sánh thêm ở cùng tỷ lệ
tile giữ lại. Nếu cần dò vùng chi tiết, thay danh sách threshold và giữ tên output
riêng cho mỗi lần đo.

Ví dụ gán hai giá trị sau khi xem kết quả; 0.5 dưới đây chỉ là giá trị minh họa:

```bash
THR_LTS=0.5
THR_AVG=0.5
```

Threshold được thay khi evaluation nên không cần train lại. Đổi `PW`, `LAMBDA`
hoặc feature stage thì phải train một run mới. Có thể thử `PW` 1/2/5 trên seed 0,
đổi `RECIPE` tương ứng và lặp bước 5–6. Khi đã chọn recipe, giữ nguyên nó ở ba seed.

Control giữ toàn bộ tile để kiểm tra pipeline:

```bash
python scripts/benchmark_lts.py configs/eval/lts_kayuan.yaml \
  detector.device=cuda:0 detector.weights="$PAIR/detector.pt" \
  tile_selector.lts.weights="$PAIR/lts.pt" \
  tile_selector.lts.feature_stage="$FEATURE_STAGE" \
  tile_selector.param_free_avg.feature_stage="$FEATURE_STAGE" \
  tile_selector.lts.force_keep_all=true \
  tile_selector.param_free_avg.force_keep_all=true \
  eval.seed=0 eval.profile_stages=false eval.visualize=0 \
  eval.experiment_id="${RECIPE}_val_keepall" \
  eval.out="$RESULT_ROOT/val_seed0_keepall.json"
```

Prediction phải khớp uniform trong sai số số học. Coarse pass vẫn tốn thời gian.

## 7. Train joint hai seed còn lại với recipe đã chọn

Nếu đổi recipe khi tuning, bước 5 phải đã hoàn thành seed 0 với recipe cuối cùng.
Sau đó:

```bash
for SEED in 42 100; do
  python scripts/train_lts.py configs/train/lts_kayuan.yaml \
    detector.weights="$RUN_ROOT/yolo26m_p2_tiled_${ATTN}_seed${SEED}/weights/best.pt" \
    features.feature_stage="$FEATURE_STAGE" \
    training_mode=joint train.seed="$SEED" train.device=cuda:0 \
    train.pos_weight="$PW" train.lambda_rel="$LAMBDA" \
    out="$RUN_ROOT/lts_joint_${RECIPE}_seed${SEED}" || break
done
```

## 8. Đánh giá test cho ba seed

Sau khi chốt recipe và threshold trên validation, chạy test trên **ảnh gốc**.
Evaluator tạo tile khi inference; không dùng ảnh test đã cắt để tính FPS ảnh gốc.

```bash
for SEED in 0 42 100; do
  PAIR=$(pair_for_seed "$SEED") || break
  python scripts/benchmark_lts.py configs/eval/lts_kayuan.yaml \
    dataset.images="$DATA_ROOT/test/images" \
    dataset.labels="$DATA_ROOT/test/labels" \
    detector.device=cuda:0 detector.weights="$PAIR/detector.pt" \
    tile_selector.lts.weights="$PAIR/lts.pt" \
    tile_selector.lts.feature_stage="$FEATURE_STAGE" \
    tile_selector.param_free_avg.feature_stage="$FEATURE_STAGE" \
    tile_selector.lts.threshold="$THR_LTS" \
    tile_selector.param_free_avg.threshold="$THR_AVG" \
    eval.seed="$SEED" eval.profile_stages=false eval.visualize=0 \
    eval.experiment_id="${RECIPE}_test" \
    eval.out="$RESULT_ROOT/test_seed${SEED}.json" || break
done
```

Mỗi file chứa ba row: **uniform**, **param_free_avg**, **lts**, dùng cùng detector
joint của seed đó. Đây là phép so sánh ảnh hưởng của selection trên một detector.

Để đo riêng baseline ảnh nguyên và uniform trước joint:

```bash
for SEED in 0 42 100; do
  for BASELINE in notile tiled; do
    TILE_TYPE=uniform
    if [ "$BASELINE" = notile ]; then TILE_TYPE=none; fi
    python scripts/benchmark_lts.py configs/eval/lts_kayuan.yaml \
      dataset.images="$DATA_ROOT/test/images" \
      dataset.labels="$DATA_ROOT/test/labels" \
      tiling.type="$TILE_TYPE" 'tile_selector.type=[none]' \
      detector.device=cuda:0 \
      detector.weights="$RUN_ROOT/yolo26m_p2_${BASELINE}_${ATTN}_seed${SEED}/weights/best.pt" \
      eval.seed="$SEED" eval.profile_stages=false eval.visualize=0 \
      eval.experiment_id="${ATTN}_${BASELINE}_prejoint_test" \
      eval.out="$RESULT_ROOT/${BASELINE}_prejoint_test_seed${SEED}.json" || break 2
  done
done
```

## 9. Tổng hợp mean +/- std của ba seed

```bash
python scripts/collect_e02.py \
  "$RESULT_ROOT/test_seed0.json" \
  "$RESULT_ROOT/test_seed42.json" \
  "$RESULT_ROOT/test_seed100.json" \
  --out "$RESULT_ROOT/test_summary.json"
```

Output: `test_summary.json` với mean/std dạng số và `test_summary.csv` với ô
`mean +/- std`. Std là sample standard deviation giữa ba seed (`ddof=1`).

Nếu đã chạy cả hai baseline ở bước 8, có thể tạo thêm một bảng đủ năm nhóm:

```bash
python scripts/collect_e02.py \
  "$RESULT_ROOT"/notile_prejoint_test_seed{0,42,100}.json \
  "$RESULT_ROOT"/tiled_prejoint_test_seed{0,42,100}.json \
  "$RESULT_ROOT"/test_seed{0,42,100}.json \
  --out "$RESULT_ROOT/all_methods_summary.json"
```

Năm nhóm là notile, uniform trước joint, uniform sau joint, param-free sau joint
và LTS sau joint. CSV giữ `experiment_id` để nhận biết hai row uniform khác nhau.
So uniform sau joint với LTS để tách ảnh hưởng của selection; so uniform trước/sau
joint để xem ảnh hưởng của training bổ sung.

Không dùng glob `test_seed*.json`, vì nó còn bắt các file `.predictions.json`.
Collector yêu cầu cùng cấu hình, evaluator, hardware và source code giữa các seed.

## 10. Timing từng bước và visualization

FPS ở bước 8 dùng ít instrumentation hơn. Để đo breakdown, chạy lại loop test
với `eval.profile_stages=true`, `eval.experiment_id="${RECIPE}_test_profile"`
và `eval.out="$RESULT_ROOT/profile_test_seed${SEED}.json"`. Tổng hợp các file
profile thành một report riêng bằng cùng collector.

Với visualization trên validation, chạy seed 0 như bước 6 nhưng đặt
`eval.visualize=3` và tên output mới. Heatmap, GT và tile được giữ sẽ được ghi
vào các thư mục hình cạnh file JSON.

| Metric | Ý nghĩa |
|---|---|
| mAP50, mAP50-95, AP_S | COCO bbox AP trên tọa độ/diện tích ảnh gốc |
| Precision, Recall, F1 | Micro matching IoU 0.5 tại conf được ghi trong config |
| tiles_per_img, kept_per_img | Số tile ứng viên và tile chạy YOLO trên một ảnh gốc |
| FPS | Số ảnh gốc / wall_s |
| latency_mean_ms | 1000 * wall_s / số ảnh gốc |
| wall_s | Tổng thời gian pipeline; ảnh đã được đọc sẵn, warmup được loại khỏi phép đo |
| select_s | Coarse preprocess/backbone, relevance và chọn tile |
| detect_s | Chuẩn bị crop, fine YOLO và remap |
| merge_s | NMS gộp kết quả trên ảnh gốc |
| time_min | wall_s / 60, tức thời gian evaluation |

Template dùng `detector.conf=0.001` cho AP; P/R cũng ở conf này. Nếu cần P/R tại
0.25 như thí nghiệm cũ, đo một lượt riêng với `detector.conf=0.25` và output mới.

## Thêm GridNet và TileCNN

Hai baseline này vẫn dùng `scripts/train_selector.py`. GridNet học từ ảnh gốc;
TileCNN dùng các tile train/val được tạo ở bước 3. Ví dụ cho seed 0:

```bash
python scripts/train_selector.py configs/train/selector_kayuan.yaml \
  type=gridnet gridnet.seed=0 \
  out=weights/lat_yolo/kayuan/gridnet_seed0.pth

python scripts/train_selector.py configs/train/selector_kayuan.yaml \
  type=tile_cnn tile_cnn.seed=0 \
  tile_cnn.train_images="$DATA_ROOT/train_lat640_ov20_v1/images" \
  tile_cnn.train_labels="$DATA_ROOT/train_lat640_ov20_v1/labels" \
  tile_cnn.val_images="$DATA_ROOT/val_lat640_ov20_v1/images" \
  tile_cnn.val_labels="$DATA_ROOT/val_lat640_ov20_v1/labels" \
  out=weights/lat_yolo/kayuan/tilecnn_seed0.pth
```

Chọn hyperparameter/threshold của từng baseline trên validation. Lặp seed 42/100
với tên weight riêng. Khi evaluate, dùng loop bước 8 với:

```text
tile_selector.type=[none,tile_cnn,gridnet,param_free_avg,lts]
tile_selector.gridnet.weights=weights/lat_yolo/kayuan/gridnet_seed${SEED}.pth
tile_selector.tile_cnn.weights=weights/lat_yolo/kayuan/tilecnn_seed${SEED}.pth
```

Thêm các threshold đã chọn vào lệnh, dùng một experiment ID và bộ output mới.
Như vậy cả năm selector dùng cùng checkpoint fine detector của từng seed.

## Đổi dataset

Trong trainer detector đổi `dataset.data_yaml`. Trong trainer LTS đổi
`dataset.data_yaml`, `train_images`, `train_labels`, `val_images`, `val_labels`,
`split_manifest`. Trong evaluator đổi `dataset.name`, `data_yaml`, `images`,
`labels`, `split_manifest`. Đổi `RUN_ROOT`, `RESULT_ROOT` và checkpoint tương ứng.
`DATA_ROOT` ở các lệnh trên không tự thay các đường dẫn mặc định trong config LTS.

Chi tiết phương pháp và trạng thái kiểm chứng:
[lat_yolo_implementation.md](lat_yolo_implementation.md).
