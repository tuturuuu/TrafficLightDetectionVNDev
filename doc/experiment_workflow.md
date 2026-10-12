# Quy trình experiments: YOLO + CBAM, tiling, CNN và GridNet

Hướng dẫn theo source sau refactor `6dcbd5c1`. Chạy từ thư mục gốc repository.
Các lệnh bên dưới là hướng dẫn, không được tự động chạy khi mở tài liệu.

## 1. Thiết kế thí nghiệm

| Phương pháp | Dữ liệu train detector | Detector khi test | Selector khi test |
|---|---|---|---|
| YOLO + CBAM, no tile | Ảnh gốc | Checkpoint no-tile | `none`, không cắt tile |
| YOLO + CBAM, uniform | Tile offline của train/val | Checkpoint tiled | `none`, giữ tất cả tile |
| YOLO + CBAM, proposal CNN | Giống uniform | Cùng checkpoint tiled | `tile_cnn` |
| YOLO + CBAM, GridNet | Giống uniform | Cùng checkpoint tiled | `gridnet` |

CNN và GridNet là hai phương pháp thay thế, không phải chạy nối tiếp CNN rồi GridNet.
Không cần train lại YOLO cho từng selector. Mỗi detector train với seed 0/42/100.
Quy trình dưới đây train selector một lần cho mỗi cấu hình, rồi dùng chung với ba
YOLO seed: độ lệch chuẩn thu được chỉ phản ánh biến thiên giữa các YOLO seed.

Chốt split trước khi train. Với ảnh từ video, ưu tiên tách theo video/sequence,
không chia ngẫu nhiên từng frame. Split Kayuan hiện tại đã được README cảnh báo
về leakage; các lệnh dưới đây không sửa split đó. Đổi split thì phải train lại.
Train dùng để học; validation dùng chọn hyperparameter; test chỉ đánh giá cuối.

## 2. Môi trường và dataset

```bash
conda activate traffic_light
pip install -e .
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

GPU vật lý 1 trở thành GPU 0 trong process. Các lệnh detector bên dưới dùng
`train.device=0`, eval dùng `detector.device=0`; selector tự dùng CUDA đang visible.
Không chạy song song các experiment đo speed trên cùng GPU.
Nếu môi trường thiếu dependency, cài `requirements.txt` theo README trước.

Chọn một dataset, chạy từ đầu đến cuối, rồi đổi dataset và lặp lại:

```bash
DS=kayuan
```

Hoặc `DS=seadronesee`. Khối sau thiết lập đúng hai layout khác nhau:

```bash
case "$DS" in
  kayuan)
    ROOT=data/kayuan
    TRAIN_IMAGES="$ROOT/train/images"; TRAIN_LABELS="$ROOT/train/labels"
    VAL_IMAGES="$ROOT/val/images"; VAL_LABELS="$ROOT/val/labels"
    TEST_IMAGES="$ROOT/test/images"; TEST_LABELS="$ROOT/test/labels"
    ;;
  seadronesee)
    ROOT=data/SeaDroneSees_70_15_15
    TRAIN_IMAGES="$ROOT/images/train"; TRAIN_LABELS="$ROOT/labels/train"
    VAL_IMAGES="$ROOT/images/val"; VAL_LABELS="$ROOT/labels/val"
    TEST_IMAGES="$ROOT/images/test"; TEST_LABELS="$ROOT/labels/test"
    ;;
  *) printf 'Chua khai bao dataset: %s\n' "$DS"; return 1 2>/dev/null || exit 1 ;;
esac
DATA_YAML="$ROOT/data.yaml"
TAG=cbam_cmp_v1
TILE=640
OVERLAP=0.2
SUFFIX="_tiled_${TAG}_t${TILE}"
DET_CFG=configs/train/detector_kayuan.yaml
SEL_CFG=configs/train/selector_kayuan.yaml
EVAL_CFG=configs/eval/kayuan.yaml
OUT="outputs/$DS/$TAG"
CNN="weights/${DS}_${TAG}_cnn_pw2.pth"
mkdir -p "$OUT/val" "$OUT/test"
```

Ở đây dùng config Kayuan làm template chung và ghi đè mọi đường dẫn liên quan.
SeaDroneSee vẫn train đúng class vì `names` được đọc từ `DATA_YAML`, không từ
tên template. Với dataset khác, thêm nhánh `case` chứa ROOT và sáu thư mục ảnh/label.
VisDrone/Bosch cần chuẩn bị annotation YOLO và class mapping trước; không dùng
nguyên annotation native của chúng. Mỗi dataset train detector/selector riêng.

`TAG` phải mới nếu không muốn dùng lại/ghi đè run cũ. `SUFFIX` riêng không đụng
`train_tiled/`, `val_tiled/` hay `_tiled_uniform/` hiện có.
Dataset YAML cần `train`, `val`, `test`, `names`; mỗi label là
`class_id x_center y_center width height` với tọa độ chuẩn hóa.
Trong code prepare-detector hiện tại, đường dẫn split được nối với thư mục chứa
YAML; nên dùng các đường dẫn split tương đối trực tiếp với thư mục đó, như hai
dataset hiện có. Code này chưa xử lý đầy đủ YAML có `path:` ở một root khác.
Eval hiện đọc file `.jpg`/`.png` chữ thường trực tiếp trong thư mục; không đọc
đệ quy hoặc `.jpeg`. Kiểm tra số ảnh/label để tránh bỏ sót dữ liệu.

## 3. Train baseline YOLO + CBAM, không tile

```bash
python scripts/train_detector.py "$DET_CFG" \
  dataset.name="$DS" dataset.data_yaml="$DATA_YAML" \
  model.arch=yolov8m model.attention=cbam \
  tiling.type=none 'train.seeds=[0,42,100]' train.device=0 \
  output.name="${DS}_${TAG}_notile"
```

Thêm `--dry-run` trước khi chạy thật để kiểm tra tham số. Mặc định template:
epochs=100, batch=16, imgsz=640, patience=30; `model.pretrained=null` không gọi
load checkpoint khởi tạo. Giữ cùng cách khởi tạo, optimizer và các tham số train
cho no-tile và tiled. Nếu dùng pretrained, đặt cùng `model.pretrained` cho cả hai.

Checkpoint:
`runs/<DS>_<TAG>_notile_seed{0,42,100}/weights/best.pt`.
Đây là detector thứ nhất; không lấy checkpoint này làm detector adaptive.

## 4. Train YOLO + CBAM trên tile uniform

```bash
python scripts/train_detector.py "$DET_CFG" \
  dataset.name="$DS" dataset.data_yaml="$DATA_YAML" \
  model.arch=yolov8m model.attention=cbam \
  tiling.type=uniform tiling.tile_size="$TILE" tiling.overlap="$OVERLAP" \
  tiling.suffix="$SUFFIX" 'train.seeds=[0,42,100]' train.device=0 \
  output.name="${DS}_${TAG}_tiled"
```

Script tự tạo train/val tile cùng YOLO label được clip, và ghi
`<ROOT>/data<SUFFIX>.yaml`. Mặc định `min_visibility=0.3`: bỏ fragment nhỏ hơn
30% diện tích GT gốc. Tập test vẫn giữ ảnh/GT gốc, không cần cắt offline.
Thư mục `train<SUFFIX>/` và `val<SUFFIX>/` được dùng lại nếu marker xác nhận
tham số giống nhau; đổi tham số thì đổi suffix. Tile cũ không có marker chỉ
được dùng lại kèm cảnh báo, không phải đã kiểm chứng tham số.

Checkpoint:
`runs/<DS>_<TAG>_tiled_seed{0,42,100}/weights/best.pt`.
Đây là detector dùng chung cho uniform, proposal CNN và GridNet.

## 5. Train proposal CNN trên tile

```bash
python scripts/train_selector.py "$SEL_CFG" \
  type=tile_cnn out="$CNN" \
  tile_cnn.train_images="$ROOT/train$SUFFIX/images" \
  tile_cnn.train_labels="$ROOT/train$SUFFIX/labels" \
  tile_cnn.val_images="$ROOT/val$SUFFIX/images" \
  tile_cnn.val_labels="$ROOT/val$SUFFIX/labels" \
  tile_cnn.pos_weight=2 tile_cnn.epochs=40
```

CNN học tile có ít nhất một box hay không. `pos_weight=2` là điểm khởi đầu,
không phải lựa chọn tối ưu cho mọi dataset. Nếu cần so pw2/pw5, train checkpoint
riêng với `tile_cnn.pos_weight=5` và `out=...cnn_pw5.pth`, rồi sweep validation
như bước 7. Không gán pw GridNet cho CNN một cách mặc định.

Lưu ý source hiện tại: training CNN vẫn gọi script legacy; early stopping và
scheduler dựa trên training loss, checkpoint là model cuối, không phải best
validation-loss. Script in sẵn tile-level recall/reduction theo nhiều threshold,
nhưng vẫn cần bước 7 để đo detection trên ảnh gốc. Trước một nghiên cứu mới cần
quy trình checkpoint CNN dựa trên validation và kiểm soát seed được triển khai
hoặc ghi rõ hạn chế này trong phương pháp; refactor chưa giải quyết phần đó.

## 6. Train GridNet trên ảnh gốc

Ví dụ so hai ứng viên pw5/pw8. Có thể thêm 10 vào vòng lặp, không thay thủ công:

```bash
for PW in 5 8; do
  python scripts/train_selector.py "$SEL_CFG" \
    type=gridnet out="weights/${DS}_${TAG}_grid_pw${PW}.pth" \
    gridnet.images="$TRAIN_IMAGES" gridnet.labels="$TRAIN_LABELS" \
    gridnet.grid_rows=8 gridnet.grid_cols=8 gridnet.epochs=40 \
    gridnet.pos_weight="$PW" gridnet.threshold=0.1 || break
done
```

GridNet không dùng ảnh đã tile. `gridnet.threshold` lúc train chỉ tính
cell-recall/keep-frac; không thay gradient hay tiêu chí chọn checkpoint.
Checkpoint chọn theo validation loss thấp nhất trong từng lần train.
Không so trực tiếp weighted BCE loss của pw5 với pw8 để chọn pw tốt hơn.

Hiện source tự tách 10% ảnh TRAIN làm validation nội bộ, chưa có tham số
external validation hoặc training seed đầy đủ. Đường dẫn VAL ở bước 7 là
validation độc lập để chọn cấu hình pipeline. Split nội bộ vẫn theo ảnh, không
theo sequence; muốn kiểm soát sequence/seed đầy đủ cần bổ sung source trước.
`keep-frac` là tỷ lệ ô grid, không phải tỷ lệ tile thực sự giữ khi có overlap.
GridNet đánh dấu tâm mọi GT box, chưa có điều kiện riêng theo kích thước small.

## 7. Chọn threshold và pos-weight trên validation

Thiết lập tham số chung cho mọi phương pháp:

```bash
CONF=0.25
COMMON=(dataset.name="$DS" tiling.tile_size="$TILE" tiling.overlap="$OVERLAP"
        detector.conf="$CONF" detector.imgsz=640 detector.batch=16 detector.device=0
        merger.iou_thr=0.5 'eval.metrics=[map50,map50_95,precision,recall,fps]')
VAL_COMMON=("${COMMON[@]}" dataset.images="$VAL_IMAGES" dataset.labels="$VAL_LABELS"
            detector.weights="runs/${DS}_${TAG}_tiled_seed0/weights/best.pt")
```

Sweep proposal CNN. Mỗi JSON có uniform và tile_cnn:

```bash
for THR in 0.01 0.02 0.05 0.1 0.2 0.3 0.5; do
  python scripts/evaluate.py "$EVAL_CFG" "${VAL_COMMON[@]}" \
    'tile_selector.type=[none,tile_cnn]' tile_selector.tile_cnn.weights="$CNN" \
    tile_selector.tile_cnn.threshold="$THR" \
    eval.out="$OUT/val/cnn_pw2_thr${THR}_seed0.json" || break
done
```

Sweep GridNet. Mỗi JSON có uniform và gridnet:

```bash
for PW in 5 8; do
  for THR in 0.01 0.02 0.05 0.1 0.2; do
    python scripts/evaluate.py "$EVAL_CFG" "${VAL_COMMON[@]}" \
      'tile_selector.type=[none,gridnet]' \
      tile_selector.gridnet.weights="weights/${DS}_${TAG}_grid_pw${PW}.pth" \
      tile_selector.gridnet.threshold="$THR" \
      eval.out="$OUT/val/grid_pw${PW}_thr${THR}_seed0.json" || break 2
  done
done
```

Chốt tiêu chí TRƯỚC khi xem test: ví dụ chỉ xét cấu hình làm giảm Recall không
quá 0.5 điểm phần trăm và giảm mAP50 không quá 0.5 điểm phần trăm so với uniform
cùng checkpoint/tile size; trong nhóm thỏa điều kiện, chọn wall_s nhỏ nhất.
Đây là ví dụ guardrail, có thể đổi theo yêu cầu nghiên cứu, không phải kết luận
rằng mọi dataset đều chấp nhận mất recall như nhau.
Nếu không ứng viên nào đạt, hạ threshold/thay huấn luyện hoặc báo rõ trade-off.

Ban đầu dùng seed0 để giảm chi phí sweep; nên kiểm tra cấu hình được chọn trên
validation seed42/100 trước khi khóa lựa chọn. Không chỉnh threshold theo test.
CNN và GridNet có threshold riêng; không bắt buộc cùng giá trị.
Không lấy threshold Kayuan tối ưu làm mặc định tối ưu cho SeaDroneSee.

Nếu sweep tile size, cũng làm trên validation, ví dụ override
`'tiling.tile_size=[640,800,960]'`. So selector với uniform ở CÙNG tile size.
GridNet có thể dùng một checkpoint cho nhiều tile size; CNN train trên tile640
có thể lệch phân phối nếu test tile khác. Giữ một tile size đã chốt khi đo test.

## 8. Test cuối cùng, ba seed và bốn phương pháp

Điền các giá trị đã chọn ở bước 7. Dưới đây chỉ là ví dụ, không phải kết quả
tối ưu đã được xác nhận cho dataset đang chọn:

```bash
GRID_PW=8
GRID_THR=0.2
CNN_THR=0.05
GRID="weights/${DS}_${TAG}_grid_pw${GRID_PW}.pth"
```

Chạy no-tile và ba phương pháp tiled trên cùng tập ảnh gốc, cùng seed:

```bash
for SEED in 0 42 100; do
  python scripts/evaluate.py "$EVAL_CFG" "${COMMON[@]}" \
    dataset.images="$TEST_IMAGES" dataset.labels="$TEST_LABELS" \
    tiling.type=none tile_selector.type=none \
    detector.weights="runs/${DS}_${TAG}_notile_seed${SEED}/weights/best.pt" \
    eval.out="$OUT/test/notile_seed${SEED}.json" || break

  python scripts/evaluate.py "$EVAL_CFG" "${COMMON[@]}" \
    dataset.images="$TEST_IMAGES" dataset.labels="$TEST_LABELS" \
    tiling.type=uniform 'tile_selector.type=[none,tile_cnn,gridnet]' \
    detector.weights="runs/${DS}_${TAG}_tiled_seed${SEED}/weights/best.pt" \
    tile_selector.tile_cnn.weights="$CNN" tile_selector.tile_cnn.threshold="$CNN_THR" \
    tile_selector.gridnet.weights="$GRID" tile_selector.gridnet.threshold="$GRID_THR" \
    eval.out="$OUT/test/tiled_selectors_seed${SEED}.json" || break
done
```

Mỗi `notile_seed*.json` có một row `notile`.
Mỗi `tiled_selectors_seed*.json` có ba row `uniform`, `tile_cnn`, `gridnet`.
Mỗi JSON kèm YAML ghi config thực tế. Không dùng lại file kết quả tên cũ;
thay TAG hoặc eval.out khi chạy experiment khác.

Nếu chỉ muốn kiểm tra pipeline, thêm `eval.max_images=20` và đặt output tên
`smoke_...`; không dùng kết quả smoke làm bảng test. Eval không có dry-run.
Không có bước auto-resume: nếu một lệnh lỗi, kiểm tra các file đã hoàn thành
trước khi chạy lại vòng lặp, vì source chủ động từ chối overwrite.

## 9. Metrics và báo cáo

Lấy metrics của cả bốn phương pháp từ JSON TEST của `scripts/evaluate.py`,
không dùng mAP trên val tile trong training CSV làm số test ảnh gốc.

| Metric | Ý nghĩa / cách lấy |
|---|---|
| `tiles_per_img` | Số tile ban đầu trung bình trên một ảnh gốc |
| `kept_per_img` | Số tile trung bình được đưa vào YOLO |
| `mAP50`, `mAP50-95` | AP có xét class, evaluator riêng của adaptile |
| `precision`, `recall` | Micro P/R tại IoU 0.5 và confidence đã cấu hình |
| `fps` | Số ảnh gốc / `wall_s`, không phải tile FPS |
| `select_s` | Tổng thời gian selector và ánh xạ mask |
| `detect_s` | Crop + detector + chuyển tọa độ, trước final merge |
| `wall_s` | Tổng pipeline, bao gồm select/detect/final merge và overhead |
| Latency trung bình | `1000 * wall_s / N` ms/ảnh; N là số ảnh thật được xử lý |
| Time test | `wall_s / 60` phút |

`wall_s` đã chứa `select_s` và `detect_s`, không cộng cả ba.
Timing không gồm đọc/decode ảnh từ đĩa, load model, warmup, tính accuracy hay
ghi JSON. Đây là pipeline timing trên ảnh đã nằm trong RAM, không phải full
disk-to-output end-to-end. JSON chưa lưu N; giữ console log/split manifest để
có đúng mẫu số. `1000 / fps` cho latency xấp xỉ vì fps trong JSON đã làm tròn.

Cho mỗi dataset và phương pháp: tính metric từng seed rồi báo mean +/- sample
SD (ddof=1, n=3). Speedup phải tính cặp uniform/selector của cùng seed trước
khi tổng hợp. Không gộp dữ liệu nhiều dataset thành một mean duy nhất.
Nếu cần đánh giá độ ổn định speed, chạy lặp lại với tên output khác và báo rõ
số lần; SD qua YOLO seed không thay cho độ dao động timing qua nhiều lần chạy.

`CONF=0.25` giữ cách đo operating point của experiments trước. Nhưng predictions
dưới 0.25 đã bị lọc, nên AP không dùng toàn bộ score range. Nếu paper cần AP với
confidence floor thấp hoặc benchmark COCO/Ultralytics chính thức, chốt protocol
đó từ validation rồi áp dụng cho mọi phương pháp; không trộn AP ở conf0.25 với
AP ở conf0.001. Evaluator adaptile có class-aware AP101 nhưng không bảo đảm
giống hoàn toàn COCO/Ultralytics matching/interpolation/P-R convention.

Selector hiện được train để tìm vật thể nói chung, không chỉ small object.
Nếu claim chính là small-object detection, cần định nghĩa small theo dataset
và bổ sung đánh giá AP/Recall riêng theo kích thước; JSON hiện chưa có AP_small.

`scripts/collect_report_metrics.py` vẫn dùng training CSV và optional validation
speed của Ultralytics, không tổng hợp bộ JSON pipeline này. Dùng nó cho bảng
huấn luyện riêng; `time_min` trong bảng đó là thời gian TRAIN, không phải test.

## 10. Thứ tự thực hiện ngắn gọn

1. Chốt split, chuẩn hóa YOLO và khai báo biến dataset.
2. Train no-tile 3 seed.
3. Train tiled 3 seed, tự tạo tile train/val.
4. Train CNN trên tile và GridNet trên ảnh gốc của dataset đó.
5. Sweep CNN threshold và GridNet pos-weight/threshold trên validation.
6. Khóa cấu hình, test cùng ảnh gốc cho 4 phương pháp x 3 YOLO seed.
7. Tổng hợp test metrics mean +/- SD; giữ JSON, config, split và log.

Không chạy lại bước train nếu checkpoint đã phù hợp split/protocol và đủ ba
seed: dùng đường dẫn checkpoint thật ở bước eval. Refactor structure tự nó
không yêu cầu train lại; thay split hoặc cấu hình train mới có thể yêu cầu.
