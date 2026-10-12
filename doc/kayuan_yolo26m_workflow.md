# Kayuan: YOLO26m-P2 + CBAM, uniform, proposal CNN và GridNet

Kiểm tra theo source hiện tại ngày 2026-10-12. Các lệnh trong tài liệu không
được tự chạy. Thực hiện từng bước, chỉ sang bước sau khi bước trước thành công.

## 1. Protocol và dữ liệu

Giữ hướng thí nghiệm đã yêu cầu: thay YOLOv8 bằng YOLO26m, vẫn dùng CBAM.
Trong repo, `model.arch=yolo26m model.attention=cbam` chọn
`configs/models/yolo26m-cbam.yaml`, có bốn mức đầu ra P2/P3/P4/P5. Vì vậy ghi
trong report là **YOLO26m-P2 + CBAM**, không phải YOLO26m stock.
Nếu cần detector không attention, đặt attention=none cho cả no-tile và tiled;
không chỉ đổi một bên rồi quy mọi thay đổi metric cho tiling.

Ảnh gốc vẫn là Kayuan, nhưng dùng split mới `data/kayuan_sequence` (symlink
tới nguồn), không train trên split frame cũ `data/kayuan`. Split mới theo nhóm
video: 14127 train / 3036 val / 3035 test theo audit. Không xóa/di chuyển ảnh
nguồn vì symlink phụ thuộc chúng. Đọc `doc/handoff_3_2_3_4.md` để tái tạo split
nếu sang máy khác. Không dùng weight cũ đã từng thấy frame của test mới.

| Phương pháp | Detector | Tile selector khi eval |
|---|---|---|
| No-tile | Train YOLO26m-P2 + CBAM trên ảnh gốc | none, không cắt tile |
| Uniform | Train YOLO26m-P2 + CBAM trên tile | none, giữ toàn bộ tile |
| Proposal CNN | Cùng checkpoint detector uniform | tile_cnn |
| GridNet | Cùng checkpoint detector uniform | gridnet |

CNN và GridNet là hai nhánh độc lập. Không train lại detector cho mỗi selector.
Recipe: scratch, 100 epoch, imgsz640, batch16, patience30, seed0/42/100;
tile640, overlap0.2, minimum visibility0.3. Chọn threshold trên val, test sau
khi khóa cấu hình. Các giá trị này là recipe khởi đầu, không cam kết tối ưu.

## 2. Môi trường và tên output

Chạy từ gốc repo, trong cùng phiên Bash để giữ biến:

```bash
conda activate traffic_light
pip install -e .
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
ROOT=data/kayuan_sequence
RUN=runs/kayuan26_cbam_v1
OUT=outputs/kayuan/yolo26_cbam_v1
CNN=weights/kayuan26_cnn_v1.pth
GRID=weights/kayuan26_grid_pw8_v1.pth
DET_CFG=configs/experiments/kayuan_p2/A6_train.yaml
SEL_CFG=configs/train/selector_kayuan.yaml
EVAL_CFG=configs/eval/kayuan.yaml
```

Chọn GPU đang rảnh; GPU vật lý 1 trở thành GPU0 trong process. Config dùng
device0. Nếu file/run đã tồn tại, đổi suffix v1 thành v2 thay vì overwrite.
Lệnh detector dưới đây lấy A6 làm template CBAM cho cả no-tile và tiled.

## 3. Train no-tile, ba seed

```bash
python scripts/train_detector.py "$DET_CFG" \
  dataset.data_yaml="$ROOT/data.yaml" \
  model.arch=yolo26m model.attention=cbam model.pretrained=null \
  tiling.type=none 'train.seeds=[0,42,100]' train.device=0 \
  output.project="$RUN" output.name=notile
```

Trước khi chạy thật có thể thêm `--dry-run` để in recipe. Checkpoint:
`$RUN/notile_seed{0,42,100}/weights/best.pt`. Không nối fine-tune checkpoint
no-tile sang tiled: hai nhóm train độc lập, cùng cách khởi tạo và recipe.

## 4. Train tiled, ba seed

```bash
python scripts/train_detector.py "$DET_CFG" \
  dataset.data_yaml="$ROOT/data.yaml" \
  model.arch=yolo26m model.attention=cbam model.pretrained=null \
  tiling.type=uniform tiling.tile_size=640 tiling.overlap=0.2 \
  tiling.suffix=_tiled_ablation_p2_t640 \
  'train.seeds=[0,42,100]' train.device=0 \
  output.project="$RUN" output.name=tiled
```

Tự cắt train/val hoặc dùng lại tile đã xác nhận cùng tham số, tại
`$ROOT/train_tiled_ablation_p2_t640/{images,labels}` và val tương ứng. Suffix
này giống config ablation hiện có; không đụng dataset tile trong root Kayuan cũ.
Test giữ ảnh gốc và cắt tile tại runtime. Checkpoint:
`$RUN/tiled_seed{0,42,100}/weights/best.pt`.
Nếu thay suffix, cũng sửa bốn đường dẫn train/val của CNN ở bước 5.

Nếu OOM, giảm batch ở cả hai nhóm detector cho thống nhất, dùng tên project
mới, và ghi nhận thay đổi. Không coi một recipe lỗi giữa chừng là run hoàn tất.

## 5. Train proposal CNN

```bash
python scripts/train_selector.py configs/experiments/kayuan_p2/A3_train.yaml \
  out="$CNN" tile_cnn.seed=0 tile_cnn.pos_weight=2
```

A3 dùng chính tile train/val của bước 4, nhưng lệnh này chỉ train CNN, không
load detector A2. Khi eval bên dưới CNN dùng detector tiled CBAM vừa train,
không dùng detector A2 không attention của bộ ablation cũ.
Source mới chọn best checkpoint theo val loss, có patience và seed.
Pos-weight2 là điểm khởi đầu; nếu cần pw5 thì train file riêng và so trên val.

## 6. Train GridNet và ứng viên pos-weight

Một checkpoint pw8:

```bash
python scripts/train_selector.py "$SEL_CFG" type=gridnet \
  gridnet.images="$ROOT/train/images" gridnet.labels="$ROOT/train/labels" \
  gridnet.val_images="$ROOT/val/images" gridnet.val_labels="$ROOT/val/labels" \
  gridnet.pos_weight=8 gridnet.seed=0 out="$GRID"
```

GridNet học trên ảnh gốc. Val được truyền riêng: không dùng chia ngẫu nhiên
10% train. `gridnet.threshold` lúc train chỉ in cell-recall/keep-frac, không
thay gradient hay tiêu chí best checkpoint. Cell keep-frac không phải tile
keep-frac khi có overlap. Không chọn pw bằng cách so raw weighted BCE loss
giữa các pw khác nhau.

Tùy chọn chạy thêm pw5/pw10 trước validation (không train pw8 lần nữa):

```bash
for PW in 5 10; do
  python scripts/train_selector.py "$SEL_CFG" type=gridnet \
    gridnet.images="$ROOT/train/images" gridnet.labels="$ROOT/train/labels" \
    gridnet.val_images="$ROOT/val/images" gridnet.val_labels="$ROOT/val/labels" \
    gridnet.pos_weight="$PW" gridnet.seed=0 \
    out="weights/kayuan26_grid_pw${PW}_v1.pth" || break
done
```

## 7. Sweep threshold trên validation

Tạo array tham số chung sau khi có checkpoint bước 4/5/6:

```bash
EV=(tiling.type=uniform tiling.tile_size=640 tiling.overlap=0.2
    detector.conf=0.25 detector.imgsz=640 detector.batch=16 detector.device=0
    merger.iou_thr=0.5 'eval.metrics=[map50,map50_95,precision,recall,fps]'
    tile_selector.tile_cnn.weights="$CNN" tile_selector.tile_cnn.resize=area
    tile_selector.gridnet.weights="$GRID")
```

Sweep hai selector độc lập, mỗi file có uniform làm mốc cùng detector seed0:

```bash
for KIND in tile_cnn gridnet; do
  for THR in 0.01 0.02 0.05 0.1 0.2 0.3 0.5; do
    python scripts/evaluate.py "$EVAL_CFG" "${EV[@]}" \
      dataset.images="$ROOT/val/images" dataset.labels="$ROOT/val/labels" \
      detector.weights="$RUN/tiled_seed0/weights/best.pt" \
      "tile_selector.type=[none,$KIND]" "tile_selector.$KIND.threshold=$THR" \
      eval.out="$OUT/val/${KIND}_thr${THR}_seed0.json" || break 2
  done
done
```

Nếu train thêm pw5/pw10, sweep riêng từng pw với override
`tile_selector.gridnet.weights=weights/kayuan26_grid_pw5_v1.pth` (hoặc pw10)
và output có pw trong tên, ví dụ `.../gridnet_pw5_thr0.05_seed0.json`.
Không ghi đè output pw8 khi so pw khác.

Chốt guardrail trước: ví dụ giảm Recall <=0.5 điểm phần trăm và giảm mAP50
<=0.5 điểm phần trăm so với uniform; trong các ứng viên đạt guardrail chọn
wall_s thấp nhất. Ví dụ này là mức đề xuất, không phải tiêu chuẩn bắt buộc.
Kiểm tra lựa chọn trên val seed42/100 trước khi khóa, bằng cách đổi đường dẫn
detector và tên output. Không retune threshold theo test.

## 8. Khóa cấu hình, test ba seed

Điền threshold thật từ bước 7. Không xem 0.05/0.2 bên dưới là tối ưu đã xác nhận.
Nếu chọn GridNet pw5/pw10, đổi GRID và cập nhật array EV trước khi test.

```bash
CNN_THR=0.05
GRID_THR=0.2
```

```bash
for SEED in 0 42 100; do
  python scripts/evaluate.py "$EVAL_CFG" "${EV[@]}" \
    dataset.images="$ROOT/test/images" dataset.labels="$ROOT/test/labels" \
    tiling.type=none tile_selector.type=none \
    detector.weights="$RUN/notile_seed${SEED}/weights/best.pt" \
    eval.out="$OUT/test/notile_seed${SEED}.json" || break
  python scripts/evaluate.py "$EVAL_CFG" "${EV[@]}" \
    dataset.images="$ROOT/test/images" dataset.labels="$ROOT/test/labels" \
    'tile_selector.type=[none,tile_cnn,gridnet]' \
    detector.weights="$RUN/tiled_seed${SEED}/weights/best.pt" \
    tile_selector.tile_cnn.threshold="$CNN_THR" \
    tile_selector.gridnet.threshold="$GRID_THR" \
    eval.out="$OUT/test/tiled_selectors_seed${SEED}.json" || break
done
```

Ba JSON no-tile (mỗi file một row) và ba JSON tiled (mỗi file ba row: uniform,
tile_cnn, gridnet). Config YAML được lưu kèm. Eval tự tạo thư mục output.
Giữ GPU không có job cạnh tranh khi đo timing; không chạy train đồng thời.
Smoke test: thêm `eval.max_images=20` và output tên smoke khác. Không đưa số
smoke vào bảng test. Eval không có dry-run.

## 9. Report

Mỗi phương pháp báo mean +/- sample SD qua ba YOLO seed: mAP50, mAP50-95,
Precision, Recall, FPS ảnh gốc, tiles/img, kept/img, select/detect/merge/wall_s.
JSON mới thêm latency p50/p95 và mẫu per-image, params/GFLOPs, peak GPU memory.
Latency trung bình dataset: `1000 * wall_s / N`, Time test: `wall_s / 60`.
Không gọi p50 là latency trung bình. N có thể lấy từ số mẫu total latency trong
JSON mới. Speedup tính cặp uniform/selector cùng seed trước khi tổng hợp.

Timing không gồm disk read/decode, load model, warmup và tính accuracy.
Wall đã chứa select/detect/merge; không cộng wall với các thành phần đó.
Protocol conf0.25 lọc prediction score thấp trước AP. Nếu cần AP dùng floor
thấp, chốt thêm protocol conf0.001 từ validation cho mọi phương pháp và giữ
output riêng; P/R ở operating point0.25 không trộn với P/R ở floor0.001.
Evaluator class-aware là code riêng của adaptile, không giả định trùng mọi
chi tiết evaluator COCO/Ultralytics. AP_small chưa được xuất riêng.

CNN/GridNet ở đây train seed0 và dùng chung ba YOLO seed, nên SD chỉ phản ánh
biến thiên detector. Muốn SD toàn pipeline: train mỗi selector seed0/42/100
với filename riêng, giữ hyperparameter đã khóa và eval paired cùng seed.
Không dùng bảng training CSV/validation tile thay cho bảng test ảnh gốc.

Tài liệu `doc/experiment_workflow.md` đời trước có các cảnh báo về CNN best
checkpoint và GridNet external val đã được source mới sửa; dùng hướng dẫn
này và config/source hiện tại cho experiment YOLO26m.
