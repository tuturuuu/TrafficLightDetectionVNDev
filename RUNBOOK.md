# LAT-YOLO: commands for Tables I–VI

Paper: [doc/LAT_YOLO.pdf](doc/LAT_YOLO.pdf). Run commands from the repository
root in **one Bash session**. This guide uses the small train/eval templates;
all dataset paths and experiment choices are supplied explicitly.
Start with Table II; Table I appears last because it reuses the trained models.

## 1. What you are running

| Term | Meaning here |
|---|---|
| Detector | YOLO: predicts boxes, classes and confidence scores on an image or crop. Its inference checkpoint is `best.pt`, or `detector.pt` after joint training. |
| Backbone | YOLO layers that extract features. P2/P3/P4/P5 are detection scales with strides 4/8/16/32. |
| Tile / fixed tiling | A crop of the original image. Fixed tiling runs YOLO on **every** candidate crop, then remaps boxes and merges duplicates with NMS. |
| Selector | Decides which candidate tiles reach YOLO. It does not produce final detection boxes. |
| Naive filter | `tile_cnn`: a separate classifier applied to each crop. Train it with `scripts/train_selector.py`. |
| Param-free avg. | `param_free_avg`: averages sigmoid activations across channels of the detector's coarse features. No selector training or selector weight file. |
| LTS | A learned `1×1` projection of the **same detector's** coarse backbone features. Model: `src/adaptile/models/lts.py`; trainer: `scripts/train_lts.py`; selector: `src/adaptile/selectors/lts.py`. Its saved head is `lts.pt`. GridNet is a separate baseline, not LTS. |
| Frozen pilot | Freeze all detector weights and train only the LTS projection. A short engineering check; **no final paper row**. It is optional, and joint training does not load its head automatically. |
| Joint training | Update the detector and LTS together using `L_det + lambda_rel * L_rel`. Produces a matched `detector.pt` + `lts.pt` pair. Required for the learned rows. |
| Threshold δ | Keep a tile when its maximum relevance score is **strictly greater** than δ. Lower usually keeps more tiles. This is different from detection confidence. |

LAT-YOLO inference is: original image → one coarse backbone pass → LTS scores
→ selected original-resolution crops → the shared YOLO detector → remap + NMS.

| Paper table | What the commands below produce |
|---|---|
| II | Nine component-ablation rows, separately for Kaiyuan Subset 2 and BSTLD |
| III | Four δ values plus fixed tiling, using the same final Kaiyuan SE detector |
| IV | Reuse Table II's fixed/LTS × CBAM/SE rows; add parameter differences |
| V | Stock baseline versus LAT-YOLO for YOLOv8m, YOLO11m and YOLO26m |
| VI | Baseline, fixed tiling and LAT-YOLO, separately on each aerial dataset |
| I | Stock YOLO comparisons and the final Table II LAT-YOLO row |

**Recipe used here:** detector ≤100 epochs; joint LTS ≤50 epochs; early stopping;
640 input, 640 tiles, 20% overlap; seeds 0/42/100; training from scratch.
This follows the current repository workflow, not the draft's 700-epoch recipe.
These architectures retain P2–P5; the draft's statement about removing the
deepest stage does not describe them. The PDF still contains placeholders:
these commands generate measurements, not guaranteed copies of its printed scores.

## 2. Setup once

Use an environment with the pinned dependencies. For a new environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e '.[research]'
```

The prepared dataset must have YOLO labels (`class cx cy w h`), a `data.yaml`
with class names and train/val/test image **directories**, and matching label
directories. Both `train/images` and `images/train` layouts are supported below.
For Kaiyuan use **`data/kaiyuan_cd`, the 40-class paper taxonomy**, not the
16-class `kayuan_sequence` dataset. Confirm your annotation subset and mapping.

Copy this setup. Change the dataset root and visible GPU:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0       # choose a free physical GPU; it becomes cuda:0
RUN_TAG=paper_manual_v1             # use a new tag for a new recipe/repeat
SEEDS="0 42 100"
LTS_AMP=true
LTS_AMP_DTYPE=bfloat16              # if unsupported: LTS_AMP=false

use_dataset() {
  DATASET=$1
  DATA_ROOT=$(realpath "$2")
  RUN_ROOT="runs/$RUN_TAG/$DATASET"
  RESULT_ROOT="outputs/$RUN_TAG/$DATASET"
  DATA_YAML="$DATA_ROOT/data_$RUN_TAG.yaml"
  MANIFEST="$DATA_ROOT/split_manifest.json"
  mkdir -p "$RUN_ROOT" "$RESULT_ROOT"
  local dirs
  mapfile -t dirs < <(python - "$DATA_ROOT" "$DATA_YAML" "$MANIFEST" <<'PY'
import json, sys, yaml
from pathlib import Path
root, dest, manifest = map(Path, sys.argv[1:])
data = yaml.safe_load((root / "data.yaml").read_text())
inventory, directories = {}, []
for split in ("train", "val", "test"):
    images = root / data[split]  # supplied root replaces an obsolete data.yaml "path"
    parts = list(images.parts)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    labels = Path(*parts)
    if not images.is_dir() or not labels.is_dir():
        raise SystemExit(f"Missing image/label directory: {images}, {labels}")
    data[split] = str(images)
    inventory[split] = sorted(str(p.relative_to(root)) for p in labels.glob("*.txt"))
    directories.extend((images, labels))
data["path"] = str(root)
text = yaml.safe_dump(data, sort_keys=False)
if dest.exists() and dest.read_text() != text:
    raise SystemExit(f"Different local YAML already exists: {dest}; choose a new RUN_TAG")
dest.write_text(text)
if not manifest.exists():
    manifest.write_text(json.dumps({"kind": "existing_split_inventory", "splits": inventory}, indent=2))
print("\n".join(map(str, directories)))
PY
  )
  [ "${#dirs[@]}" -eq 6 ] || return 1
  TRAIN_IMAGES=${dirs[0]}; TRAIN_LABELS=${dirs[1]}
  VAL_IMAGES=${dirs[2]}; VAL_LABELS=${dirs[3]}
  TEST_IMAGES=${dirs[4]}; TEST_LABELS=${dirs[5]}
}
use_dataset kaiyuan_cd /absolute/path/to/kaiyuan_cd
```

The helper writes a local dataset YAML with the supplied root. It preserves an
existing split manifest; if absent, it records the existing partitions.
That inventory does not verify that videos/sequences are disjoint.

Copy these small helpers once. The argument names describe every call below:

```bash
# det_train RUN_NAME ARCHITECTURE_YAML none|uniform ATTENTION [extra overrides]
det_train() {
  local name=$1 arch=$2 tiling=$3 attention=$4
  shift 4
  python scripts/train_detector.py configs/train/detector_kayuan.yaml \
    dataset.name="$DATASET" dataset.data_yaml="$DATA_YAML" \
    model.arch="$arch" model.attention="$attention" model.pretrained=null \
    tiling.type="$tiling" tiling.tile_size=640 tiling.overlap=0.2 \
    tiling.min_visibility=0.3 tiling.suffix="_${RUN_TAG}_640_ov20" \
    train.device=0 "train.seeds=[0,42,100]" train.epochs=100 \
    train.imgsz=640 train.batch=16 train.patience=30 train.workers=4 \
    output.project="$RUN_ROOT" output.name="$name" "$@"
}

# lts_train RUN_NAME SEED TILED_DETECTOR FEATURE_STAGE joint|frozen_head_pilot
lts_train() {
  local name=$1 seed=$2 detector=$3 stage=$4 mode=$5
  shift 5
  python scripts/train_lts.py configs/train/lts_kayuan.yaml \
    dataset.data_yaml="$DATA_YAML" dataset.split_manifest="$MANIFEST" \
    dataset.train_images="$TRAIN_IMAGES" dataset.train_labels="$TRAIN_LABELS" \
    dataset.val_images="$VAL_IMAGES" dataset.val_labels="$VAL_LABELS" \
    detector.weights="$detector" detector.imgsz=640 \
    features.feature_stage="$stage" features.coarse_imgsz=640 \
    tiling.tile_size=640 tiling.overlap=0.2 tiling.min_visibility=0.3 \
    training_mode="$mode" train.seed="$seed" train.device=cuda:0 \
    train.epochs=50 train.batch_size=16 train.workers=4 train.patience=10 \
    train.lr=0.001 train.weight_decay=0.0005 \
    train.lambda_rel=1 train.pos_weight=1 train.threshold=0.5 \
    train.amp="$LTS_AMP" train.amp_dtype="$LTS_AMP_DTYPE" \
    out="$RUN_ROOT/${name}_seed$seed" "$@"
}

# pair RUN_NAME SEED -> directory containing the selected detector.pt and lts.pt
pair() {
  python - "$RUN_ROOT/${1}_seed$2" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
print(run / json.loads((run / "best.json").read_text())["directory"])
PY
}

# evaluate ROW_ID SEED DETECTOR none|uniform '[SELECTORS]' [extra overrides]
# SPLIT=val for tuning; SPLIT=test for final results.
evaluate() {
  local row=$1 seed=$2 detector=$3 tiling=$4 selectors=$5
  shift 5
  local images="$VAL_IMAGES" labels="$VAL_LABELS"
  if [ "$SPLIT" = test ]; then images=$TEST_IMAGES; labels=$TEST_LABELS; fi
  python scripts/benchmark_lts.py configs/eval/lts_kayuan.yaml \
    dataset.name="$DATASET" dataset.data_yaml="$DATA_YAML" \
    dataset.split_manifest="$MANIFEST" dataset.images="$images" dataset.labels="$labels" \
    detector.weights="$detector" detector.device=cuda:0 \
    detector.imgsz=640 detector.batch=16 detector.conf=0.001 \
    "tile_selector.type=$selectors" tiling.type="$tiling" \
    tiling.tile_size=640 tiling.overlap=0.2 tiling.pad=false \
    eval.seed="$seed" eval.profile_stages=false eval.visualize=0 eval.max_images=null \
    eval.experiment_id="${DATASET}_${SPLIT}_$row" \
    eval.out="$RESULT_ROOT/$SPLIT/${row}_seed$seed.json" "$@"
}
```

All LTS dataset paths are overridden: changing `DATA_ROOT` alone would otherwise
leave the LTS template's original dataset paths active. Add `--dry-run` to a
helper call to inspect its resolved arguments without training/evaluation.

## 3. Table II — component ablation

Run this section on Kaiyuan first. For the BSTLD columns, call
`use_dataset bstld /absolute/path/to/bstld_yolo` and repeat this section,
including validation tuning. Its class map and checkpoints must be its own.

### Step 1 — train the four detectors

Each command trains all three seeds. The name tells you which table row it feeds.

```bash
# A1: no tiling, no attention; same P2–P5 architecture as the controlled ablation
det_train baseline_none configs/models/yolo26m-p2.yaml none none

# A2/A3/A4 and initialization for A5
det_train fixed_none configs/models/yolo26m-p2.yaml uniform none

# A6 and initialization for A8
det_train fixed_cbam configs/models/yolo26m-cbam.yaml uniform cbam

# A7 and initialization for A9 (LAT-YOLO)
det_train fixed_se configs/models/yolo26m-se.yaml uniform se
```

Detector checkpoints: `$RUN_ROOT/<name>_seed<seed>/weights/best.pt`.
Uniform training creates `train_${RUN_TAG}_640_ov20` and
`val_${RUN_TAG}_640_ov20` beneath the dataset root. Evaluation uses original
images and creates its own candidate tiles.

The local historical Table II inventory used **Baseline + SE** for A1.
That is a different row from the PDF's no-attention baseline above. If reusing
that inventory, label A1 “Baseline + SE” and use its no-tiling SE checkpoint.

### Step 2 — train the naive CNN and three joint LTS variants

```bash
for seed in $SEEDS; do
  # A3: independent per-tile classifier; A4 needs no selector training
  python scripts/train_selector.py configs/train/selector_kayuan.yaml \
    type=tile_cnn tile_cnn.seed="$seed" \
    tile_cnn.train_images="$DATA_ROOT/train_${RUN_TAG}_640_ov20/images" \
    tile_cnn.train_labels="$DATA_ROOT/train_${RUN_TAG}_640_ov20/labels" \
    tile_cnn.val_images="$DATA_ROOT/val_${RUN_TAG}_640_ov20/images" \
    tile_cnn.val_labels="$DATA_ROOT/val_${RUN_TAG}_640_ov20/labels" \
    out="$RUN_ROOT/tilecnn_seed$seed.pth" || break

  # A5: LTS without attention; P3 features are at layer 4
  lts_train lts_none "$seed" "$RUN_ROOT/fixed_none_seed$seed/weights/best.pt" 4 joint || break

  # A8/A9: post-attention P3 features are at layer 5
  lts_train lts_cbam "$seed" "$RUN_ROOT/fixed_cbam_seed$seed/weights/best.pt" 5 joint || break
  lts_train lts_se "$seed" "$RUN_ROOT/fixed_se_seed$seed/weights/best.pt" 5 joint || break
done
```

Example final LAT-YOLO files for seed 0:
`$RUN_ROOT/lts_se_seed0/best_epochNNNN/{detector.pt,lts.pt}`.
`best.json` identifies NNNN; `pair lts_se 0` resolves it.
Always load both files from that directory. `resume.pt` is training state.

Optional frozen pilot, **not A5/A8/A9**:

```bash
lts_train pilot_se 0 "$RUN_ROOT/fixed_se_seed0/weights/best.pt" 5 frozen_head_pilot train.epochs=5
```

### Step 3 — choose thresholds on validation

Sweep each selector against fixed tiling using its **same detector checkpoint**:

```bash
SPLIT=val
for seed in $SEEDS; do
  for thr in 0.3 0.5 0.7 0.9; do
    evaluate "A3_thr$thr" "$seed" "$RUN_ROOT/fixed_none_seed$seed/weights/best.pt" uniform '[none,tile_cnn]' \
      tile_selector.tile_cnn.weights="$RUN_ROOT/tilecnn_seed$seed.pth" \
      tile_selector.tile_cnn.threshold="$thr" || break 2
    evaluate "A4_thr$thr" "$seed" "$RUN_ROOT/fixed_none_seed$seed/weights/best.pt" uniform '[none,param_free_avg]' \
      tile_selector.param_free_avg.feature_stage=4 tile_selector.param_free_avg.threshold="$thr" || break 2
    for attention in none cbam se; do
      row=A5; stage=4
      if [ "$attention" = cbam ]; then row=A8; stage=5; fi
      if [ "$attention" = se ]; then row=A9; stage=5; fi
      PAIR=$(pair "lts_$attention" "$seed") || break 3
      evaluate "${row}_thr$thr" "$seed" "$PAIR/detector.pt" uniform '[none,lts]' \
        tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage="$stage" \
        tile_selector.lts.threshold="$thr" || break 3
    done
  done
done
```

Use this explicit policy: allow **at most 0.01 absolute loss (one percentage
point)** in both mean validation mAP50 and mean recall relative to the paired
uniform control; among eligible thresholds choose the highest mean FPS.
This is a runbook policy, not a threshold specified by the paper.
Set a different tolerance before inspecting results if your experiment requires it.

The helper below prints the measurements and returns the selected δ. It uses
all three validation seeds. If none qualifies, it stops: extend the sweep toward
lower δ or revise training on validation, then rerun under a new output name.

```bash
pick_threshold() {
  python - "$RESULT_ROOT/val" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
from statistics import mean
root, prefix, method = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
eligible = []
for threshold in (0.3, 0.5, 0.7, 0.9):
    controls, selected = [], []
    for seed in (0, 42, 100):
        rows = json.loads((root / f"{prefix}_thr{threshold}_seed{seed}.json").read_text())
        controls.append(next(r for r in rows if r["method"] == "uniform"))
        selected.append(next(r for r in rows if r["method"] == method))
    loss_ap = mean(c["mAP50"] - r["mAP50"] for c, r in zip(controls, selected))
    loss_recall = mean(c["recall"] - r["recall"] for c, r in zip(controls, selected))
    fps = mean(r["fps"] for r in selected)
    kept = mean(r["kept_per_img"] / r["tiles_per_img"] for r in selected)
    print(f"{prefix} δ={threshold}: AP loss={loss_ap:.4f}, recall loss={loss_recall:.4f}, "
          f"tiles={100*kept:.1f}%, FPS={fps:.2f}", file=sys.stderr)
    if loss_ap <= .01 + 1e-12 and loss_recall <= .01 + 1e-12:
        eligible.append((fps, -kept, threshold))
if not eligible:
    raise SystemExit(f"{prefix}: no eligible threshold; do not choose using test results")
print(max(eligible)[2])
PY
}
THR_CNN=$(pick_threshold A3 tile_cnn) &&
THR_AVG=$(pick_threshold A4 param_free_avg) &&
THR_LTS_NONE=$(pick_threshold A5 lts) &&
THR_LTS_CBAM=$(pick_threshold A8 lts) &&
THR_LTS_SE=$(pick_threshold A9 lts) &&
declare -p THR_CNN THR_AVG THR_LTS_NONE THR_LTS_CBAM THR_LTS_SE > "$RESULT_ROOT/thresholds.env"
```

Proceed only after all five selections succeed. Thresholds need no retraining.
Changing `pos_weight`, `lambda_rel` or feature stage needs a new training run.
AVG, CNN and LTS have different score distributions; give each its own δ.

### Step 4 — test: each command is one named Table II row

```bash
source "$RESULT_ROOT/thresholds.env"
SPLIT=test
for seed in $SEEDS; do
  evaluate A1 "$seed" "$RUN_ROOT/baseline_none_seed$seed/weights/best.pt" none '[none]' || break
  evaluate A2 "$seed" "$RUN_ROOT/fixed_none_seed$seed/weights/best.pt" uniform '[none]' || break
  evaluate A3 "$seed" "$RUN_ROOT/fixed_none_seed$seed/weights/best.pt" uniform '[tile_cnn]' \
    tile_selector.tile_cnn.weights="$RUN_ROOT/tilecnn_seed$seed.pth" tile_selector.tile_cnn.threshold="$THR_CNN" || break
  evaluate A4 "$seed" "$RUN_ROOT/fixed_none_seed$seed/weights/best.pt" uniform '[param_free_avg]' \
    tile_selector.param_free_avg.feature_stage=4 tile_selector.param_free_avg.threshold="$THR_AVG" || break
  PAIR=$(pair lts_none "$seed") || break
  evaluate A5 "$seed" "$PAIR/detector.pt" uniform '[lts]' \
    tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=4 tile_selector.lts.threshold="$THR_LTS_NONE" || break
  evaluate A6 "$seed" "$RUN_ROOT/fixed_cbam_seed$seed/weights/best.pt" uniform '[none]' || break
  evaluate A7 "$seed" "$RUN_ROOT/fixed_se_seed$seed/weights/best.pt" uniform '[none]' || break
  PAIR=$(pair lts_cbam "$seed") || break
  evaluate A8 "$seed" "$PAIR/detector.pt" uniform '[lts]' \
    tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=5 tile_selector.lts.threshold="$THR_LTS_CBAM" || break
  PAIR=$(pair lts_se "$seed") || break
  evaluate A9 "$seed" "$PAIR/detector.pt" uniform '[lts]' \
    tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=5 tile_selector.lts.threshold="$THR_LTS_SE" || break
done
```

| Output `$RESULT_ROOT/test/…` | Exact Table II row |
|---|---|
| `A1_seed<seed>.json` | Baseline |
| `A2_seed<seed>.json` | + Fixed tiling |
| `A3_seed<seed>.json` | + Naive filter |
| `A4_seed<seed>.json` | + Param-free avg. |
| `A5_seed<seed>.json` | + LTS (learned), no attention |
| `A6_seed<seed>.json` | + Fixed + CBAM |
| `A7_seed<seed>.json` | + Fixed + SE |
| `A8_seed<seed>.json` | + LTS + CBAM |
| `A9_seed<seed>.json` | + LTS + SE (LAT-YOLO) |

For each row, report `mAP50`, `mAP50-95`, `recall`, `fps`.
Joint training changes the detector, so A2 versus A5 includes training effects.
To isolate selection alone, use the paired uniform control, as in Table III.
An AVG-versus-LTS comparison on a shared joint detector is an additional control;
do not silently replace A4's detector.

Summarize each row separately (distinct detectors need distinct experiment IDs):

```bash
for row in A1 A2 A3 A4 A5 A6 A7 A8 A9; do
  python scripts/collect_e02.py "$RESULT_ROOT/test/${row}"_seed{0,42,100}.json \
    --out "$RESULT_ROOT/${row}_summary.json" || break
done
```

Each summary has numeric mean/sample SD in JSON and a companion CSV.

## 4. Table III — threshold trade-off

Return to the **Kaiyuan** dataset selection if you just ran BSTLD.
The previous validation sweep selected `THR_LTS_SE` for the final A9 row.
Table III reports the whole curve: evaluate all four **predeclared** thresholds
on test, with unchanged weights. Do not choose a new threshold from this curve.

```bash
use_dataset kaiyuan_cd /absolute/path/to/kaiyuan_cd
source "$RESULT_ROOT/thresholds.env"
SPLIT=test
for seed in $SEEDS; do
  PAIR=$(pair lts_se "$seed") || break
  for thr in 0.3 0.5 0.7 0.9; do
    evaluate "T3_thr$thr" "$seed" "$PAIR/detector.pt" uniform '[none,lts]' \
      tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=5 \
      tile_selector.lts.threshold="$thr" || break 2
  done
  # Extra correctness check: keep all tiles but still pay for the coarse pass
  evaluate T3_keepall "$seed" "$PAIR/detector.pt" uniform '[none,lts]' \
    tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=5 \
    tile_selector.lts.force_keep_all=true || break
done
for thr in 0.3 0.5 0.7 0.9; do
  python scripts/collect_e02.py "$RESULT_ROOT/test/T3_thr${thr}"_seed{0,42,100}.json \
    --out "$RESULT_ROOT/T3_thr${thr}_summary.json" || break
done
```

| Table III row/column | Source |
|---|---|
| δ = 0.3 / 0.5 / 0.7 / 0.9 | `method=lts` in the corresponding `T3_thr…` files |
| Fixed tiling (all) | `method=uniform` in `T3_thr0.3`; same joint detector as LTS |
| Tiles proc. (%) | `100 * kept_per_img / tiles_per_img`; equivalently `100 * (1 - tile_reduction)` |
| mAP50 / Recall / FPS | `mAP50` / `recall` / `fps` |

The forced-all LTS control should match uniform predictions within numerical
tolerance. Its extra coarse-pass latency makes it different from the fixed row.
A threshold of zero is not a substitute for `force_keep_all=true`.

## 5. Table IV — CBAM versus SE

No new training: use Kaiyuan Table II A6/A7/A8/A9 summaries.

| Table IV row | Table II result | Reference for attention ΔParams |
|---|---|---|
| Fixed / CBAM | A6 | A2, fixed without attention |
| Fixed / SE | A7 | A2 |
| LTS / CBAM | A8 | A5, LTS without attention |
| LTS / SE | A9 | A5 |

Copy mean `mAP50` and `fps` from the summaries. Calculate ΔParams in millions
using the matched no-attention reference above; this defines **attention
overhead**, including any change in LTS head size. Count the shared detector
once, even though it runs on coarse and fine inputs.

```bash
python - "$RESULT_ROOT/test" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def params(row):
    r = json.loads((root / f"{row}_seed0.json").read_text())[0]
    return r["detector_parameters_loaded"] + (r["relevance_head_parameters"] or 0)
for row, reference in (("A6", "A2"), ("A7", "A2"), ("A8", "A5"), ("A9", "A5")):
    print(f"{row}: ΔParams={(params(row)-params(reference))/1e6:.6f} M")
PY
```

## 6. Table V — change the backbone YAML

Use Kaiyuan Subset 2. An explicit `model.arch=<file.yaml>` chooses that entire
architecture; `model.attention=se` alone does not insert SE into an arbitrary YAML.

| Backbone | Stock no-tiling baseline YAML | Tiled LAT detector YAML | LTS stage |
|---|---|---|---|
| YOLOv8m | `yolov8m.yaml` (Ultralytics) | `configs/models/yolov8m-p2-se.yaml` | 5 |
| YOLO11m | `yolo11m.yaml` (Ultralytics) | `configs/models/yolo11m-p2-se.yaml` | 5 |
| YOLO26m | `yolo26m.yaml` (Ultralytics) | `configs/models/yolo26m-se.yaml` | 5 |

Run each family through detector training → joint LTS → validation selection →
test. **Retrain the LTS head for each backbone**; changing only the evaluation
YAML cannot convert a trained checkpoint into another architecture.

```bash
for family in yolov8m yolo11m yolo26m; do
  case "$family" in
    yolov8m) arch=configs/models/yolov8m-p2-se.yaml ;;
    yolo11m) arch=configs/models/yolo11m-p2-se.yaml ;;
    yolo26m) arch=configs/models/yolo26m-se.yaml ;;
  esac
  det_train "T5_${family}_baseline" "$family.yaml" none none || break
  det_train "T5_${family}_fixed_se" "$arch" uniform se || break
  for seed in $SEEDS; do
    lts_train "T5_${family}_lts" "$seed" "$RUN_ROOT/T5_${family}_fixed_se_seed$seed/weights/best.pt" 5 joint || break 2
  done
  SPLIT=val
  for seed in $SEEDS; do
    PAIR=$(pair "T5_${family}_lts" "$seed") || break 2
    for thr in 0.3 0.5 0.7 0.9; do
      evaluate "T5_${family}_thr$thr" "$seed" "$PAIR/detector.pt" uniform '[none,lts]' \
        tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=5 \
        tile_selector.lts.threshold="$thr" || break 3
    done
  done
  THR=$(pick_threshold "T5_$family" lts) || break
  printf '%s\n' "$THR" > "$RESULT_ROOT/T5_${family}_threshold.txt"
  SPLIT=test
  for seed in $SEEDS; do
    evaluate "T5_${family}_baseline" "$seed" "$RUN_ROOT/T5_${family}_baseline_seed$seed/weights/best.pt" none '[none]' || break 2
    PAIR=$(pair "T5_${family}_lts" "$seed") || break 2
    evaluate "T5_${family}_lat" "$seed" "$PAIR/detector.pt" uniform '[lts]' \
      tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=5 \
      tile_selector.lts.threshold="$THR" || break 2
  done
  for method in baseline lat; do
    python scripts/collect_e02.py "$RESULT_ROOT/test/T5_${family}_${method}"_seed{0,42,100}.json \
      --out "$RESULT_ROOT/T5_${family}_${method}_summary.json" || break 2
  done
done
```

For each family, Baseline = baseline summary mean mAP50; + LAT-YOLO = lat
summary mean mAP50; ΔmAP50 = LAT − Baseline. This compares the full LAT package
(P2–P5, SE, tiling, joint LTS) against stock YOLO. The paired uniform validation
rows isolate selection on the modified detector. Existing matching runs may be
reused after resolving their real paths; the helper names above denote fresh runs.

## 7. Table VI — three methods on each aerial dataset

Use prepared, labeled YOLO data. Run each dataset's own training and validation;
do not apply Kaiyuan weights to a different class map. This evaluates the method
across domains after training on each domain, not zero-shot transfer.

Change the two roots, then copy the block:

```bash
VISDRONE_ROOT=/absolute/path/to/visdrone
SEADRONESSEE_ROOT=/absolute/path/to/seadrone_sequence
for dataset in visdrone seadronesee; do
  root=$VISDRONE_ROOT
  if [ "$dataset" = seadronesee ]; then root=$SEADRONESSEE_ROOT; fi
  use_dataset "$dataset" "$root" || break

  # Baseline: full image, no attention
  det_train T6_baseline configs/models/yolo26m-p2.yaml none none || break
  # + Fixed tiling: every tile, no attention
  det_train T6_fixed configs/models/yolo26m-p2.yaml uniform none || break
  # Extra dependency for LAT-YOLO: SE tiled detector, then joint LTS
  det_train T6_fixed_se configs/models/yolo26m-se.yaml uniform se || break
  for seed in $SEEDS; do
    lts_train T6_lts "$seed" "$RUN_ROOT/T6_fixed_se_seed$seed/weights/best.pt" 5 joint || break 2
  done

  SPLIT=val
  for seed in $SEEDS; do
    PAIR=$(pair T6_lts "$seed") || break 2
    for thr in 0.3 0.5 0.7 0.9; do
      evaluate "T6_thr$thr" "$seed" "$PAIR/detector.pt" uniform '[none,lts]' \
        tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=5 \
        tile_selector.lts.threshold="$thr" || break 3
    done
  done
  THR=$(pick_threshold T6 lts) || break
  printf '%s\n' "$THR" > "$RESULT_ROOT/T6_threshold.txt"
  SPLIT=test
  for seed in $SEEDS; do
    evaluate T6_baseline "$seed" "$RUN_ROOT/T6_baseline_seed$seed/weights/best.pt" none '[none]' || break 2
    evaluate T6_fixed "$seed" "$RUN_ROOT/T6_fixed_seed$seed/weights/best.pt" uniform '[none]' || break 2
    PAIR=$(pair T6_lts "$seed") || break 2
    evaluate T6_lat "$seed" "$PAIR/detector.pt" uniform '[lts]' \
      tile_selector.lts.weights="$PAIR/lts.pt" tile_selector.lts.feature_stage=5 \
      tile_selector.lts.threshold="$THR" || break 2
  done
  for method in baseline fixed lat; do
    python scripts/collect_e02.py "$RESULT_ROOT/test/T6_${method}"_seed{0,42,100}.json \
      --out "$RESULT_ROOT/T6_${method}_summary.json" || break 2
  done
done
```

Each dataset has its own `outputs/$RUN_TAG/<dataset>/` directory.
`T6_baseline` → “Baseline”; `T6_fixed` → “+ Fixed tiling”;
`T6_lat` → “+ LAT-YOLO”. Copy summary means for
`mAP50`, `mAP50-95`, `AP_S`, `recall`, `fps`.

This block uses the controlled P2–P5 no-attention baseline. If reproducing a
**stock** YOLO26m baseline, change its architecture argument to `yolo26m.yaml`
and label the protocol. The draft's existing numbers have to be checked against
their original run settings before reuse.

For VisDrone, generic COCO evaluation of converted YOLO labels does not implement
the official ignored-region protocol. State the split and conversion; do not
claim official benchmark equivalence. The stock converter uses test-dev as
`test`. To report official validation images as the final evaluation set,
first reserve a separate validation partition from training for threshold tuning.

## 8. Table I — detector comparison on Kaiyuan Subset 2

```bash
use_dataset kaiyuan_cd /absolute/path/to/kaiyuan_cd
SPLIT=test
# Table V already provides stock YOLOv8m, YOLO11m and YOLO26m baselines.
# Train/evaluate the two other supported YOLO rows:
for family in yolov10m yolo12m; do
  det_train "T1_$family" "$family.yaml" none none || break
  for seed in $SEEDS; do
    evaluate "T1_$family" "$seed" "$RUN_ROOT/T1_${family}_seed$seed/weights/best.pt" none '[none]' || break 2
  done
  python scripts/collect_e02.py "$RESULT_ROOT/test/T1_${family}"_seed{0,42,100}.json \
    --out "$RESULT_ROOT/T1_${family}_summary.json" || break
done
```

| Table I method | Result to use |
|---|---|
| YOLOv8m / YOLO11m / YOLO26m | Table V's corresponding stock baseline summary |
| YOLOv10m / YOLO12m | `T1_<family>_summary.json` |
| LAT-YOLO (Ours) | Kaiyuan Table II `A9_summary.json` |
| AT-YOLO | Historical cited result; A6 is not an automatic reproduction of AT-YOLO |
| RT-DETRv2-M | No RT-DETRv2 training/evaluation adapter in this repository; requires its own verified implementation/checkpoint |

Copy `mAP50`, `mAP50-95`, `precision`, `recall` and `fps`.
The new measurements use the common evaluator below; historical printed
precision/recall/FPS must not be assumed to follow this same protocol.

## 9. Important notes when reading results

- **Use evaluation JSON, not training CSV.** Each raw JSON is a list of method
  rows. Summaries report mean ± sample SD across training seeds (`ddof=1`).
  `time_min` is evaluation time, not training time.
- **Metrics:** AP uses COCO on final boxes in original-image coordinates,
  `coco_max_dets=100`. `AP_S` uses original box area. Precision/recall use micro
  IoU 0.5 matching at `detector.conf=0.001`; they differ from Ultralytics'
  best-F1 precision/recall in training logs. Measure a different confidence
  operating point in a separately named experiment.
- **Timing:** FPS counts original images. It includes coarse selection, crop
  preparation, fine inference, remapping and NMS; disk reads and full-path
  warmup are excluded. Keep GPU, batch, precision and instrumentation matched.
  Tile reduction does not guarantee a speedup.
- **Checkpoints and repeats:** output files/runs are protected against overwrite.
  Use a new `RUN_TAG` for a changed recipe. Change the tile-cache suffix when
  changing size/overlap/visibility or the underlying dataset. To resume LTS,
  repeat the exact `lts_train` call with `--resume <run>/resume.pt`.
- **Existing weights:** use them only with their actual dataset/class map,
  feature stage, recipe and matched detector/head pair. Fresh helper names do
  not automatically point at historical artifacts.
- **Breakdown for Section V-F:** repeat a paired uniform/LTS evaluation with
  `eval.profile_stages=true` and a new row ID. Inspect `stage_latency_ms` for
  coarse backbone, relevance/selection, fine inference and NMS. Keep this
  instrumented timing separate from headline FPS.
