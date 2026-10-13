# LAT-YOLO implementation record

## Approved scope, 2026-10-12

- Work directly on `lts-applied`; do not create/switch branches or commit automatically.
- `spec.md` is the method contract. Do not request or search for the private manuscript.
- Retain YOLO26m-P2/P3/P4/P5 for all matched attention variants.
- Skip P1.1 dataset leakage auditing/resplitting: user reports this work resolved.
  This is an explicit scope waiver, not a newly executed data-integrity verification.
- Preserve existing GridNet, TileCNN, YOLO configs, weights and results.
- No long GPU training without separate authorization.

The historical P0 snapshot is in `outputs/audits/p0_f7c1db15_20261012T141320Z/`.
The decisions above supersede its pending branch/manuscript/P5 questions.

## P1.2: metrics and checkpoint provenance

An optional `metrics_coco` adapter evaluates the same final, original-image
predictions with installed pycocotools. Historical `metrics.evaluate` remains
unchanged. COCO maxDets defaults to 100, unlike the uncapped historical evaluator;
outputs must identify the backend. AP_S area uses original-image boxes, not tiles.
COCO's exact area-boundary conventions are inherited from pycocotools. The adapter
does not claim to implement VisDrone ignore-region or official benchmark handling.
Existing micro precision/recall are operating-point diagnostics, not COCO AP.

P0 checkpoint hashes/class counts/training paths remain the provenance ledger.
Old task checkpoints have no verified eligibility for final clean-split claims;
they remain useful for regression or explicitly labeled engineering smoke tests.
New LTS checkpoints must identify their exact paired detector and split manifest.

The `.gitignore` dataset rule is now root-scoped so existing local data tools
referenced by tests are visible to version control. Dataset contents are untouched.

## Phase status

| Phase | Outcome | Evidence / boundary |
|---|---|---|
| P0 | Audited previously | Commit `f7c1db15`; audit files above. Decisions superseded as listed above. |
| P1.1 | Skipped | User reports split leakage resolved. No re-audit or resplitting performed. |
| P1.2 | Implemented, CPU verified | Original-pixel COCO adapter and direct pycocotools comparison; legacy evaluator unchanged. |
| P2 | Implemented, CPU verified | Graph-aware backbone-only extraction from the fine detector instance; layer-hook parity. |
| P3 | Implemented, CPU verified | Exact LTS / parameter-free equations, overlap targets, max scoring, strict threshold. |
| P4 | Implemented, CPU verified | Shared binding, zero-tile safety, nonempty all-tiles prediction parity, paired checkpoints. |
| P5 | Implemented, CPU verified; experimental recipe | Frozen and actual joint YOLO-loss training; uninterrupted/resumed weights match exactly in synthetic tests. |
| P6 | Implemented, CPU verified | Attention parameters registered before optimizer; SE/CBAM device/dtype and serialization tests. P2-P5 retained. |
| P7 | Implemented, CPU verified | Versioned evaluation, timing stages, predictions, visualizations, provenance, three-seed collector. |
| P8 | Not run | Real-data multi-seed training/threshold selection and GPU benchmarks require approval. |
| P9 | Engineering handoff documented | Commands and limitations below. Publication-ready results/figures remain pending P8. |

No LAT-YOLO accuracy, speedup or SOTA claim has been established. CPU synthetic
tests exercise code paths; their predictions and timings are not experiment results.

## Files and contracts

- `src/adaptile/models/coarse_features.py`: fixed square RGB letterbox and a
  graph-aware pass ending at one selected backbone layer. No coarse Detect pass.
- `models/lts.py`: `Conv2d(C, 1, 1, bias=True)` returning logits; exactly `C+1`
  head parameters. Parameter-free output is `mean_c(sigmoid(F))`, not the reverse.
- `selectors/{lts,param_free_avg,relevance,relevance_utils}.py`: strict `score >
  threshold`; max over all intersecting nominal feature cells. `min_keep=0` by
  default, stable top-k only when explicitly requested. Selected crops retain
  candidate order and coordinates. Empty selection never calls YOLO on `[]`.
- `relevance_targets.py`: any GT box overlap, not box centers. Boxes are clipped;
  letterbox-only cells are excluded from BCE. Partially valid cells participate.
  These are nominal cell footprints, not the entire neural receptive field.
  All annotated classes/sizes supervise relevance, not just a size-filtered subset.
- `training_lts.py`, `scripts/train_lts.py`: paired original frame + uniformly
  sampled fine tile; head-only and joint modes, epoch-boundary resume, immutable
  best detector/head pairs. Existing detector/GridNet/TileCNN trainers unchanged.
- `evaluation_lts.py`, `experiment.py`, `metrics_coco.py`: opt-in `lat_yolo_v1`
  protocol via `scripts/evaluate.py` or `scripts/benchmark_lts.py`.
- `report_lts.py`, `scripts/collect_e02.py`: mean +/- **sample** SD (`ddof=1`),
  rejecting duplicate/missing seeds and mismatched protocol/detector comparisons.

Historical `none`, `gridnet`, `tile_cnn` configs keep their legacy protocol by
default. They can also be evaluated under `lat_yolo_v1` for a matched comparison.
Do not mix historical class-agnostic, project class-aware, and COCO AP numbers.

### Retained architecture

| Variant | YAML | Coarse default | Detect inputs |
|---|---|---|---|
| none | `configs/models/yolo26m-p2.yaml` | layer 4, P3, stride 8 | 19, 22, 25, 28 |
| CBAM | `configs/models/yolo26m-cbam.yaml` | layer 5, post-CBAM P3 | 22, 25, 28, 31 |
| SE | `configs/models/yolo26m-se.yaml` | layer 5, post-SE P3 | 22, 25, 28, 31 |

All retain P2/P3/P4/P5, strides 4/8/16/32. Attention modules sit at backbone
layers 5, 8, 11 in the attention variants. The existing YAML topology is unchanged.
The default coarse P3 tensor has 512 channels; the LTS head adds 513 parameters.
At coarse size 640 its map is 80x80, not GridNet's independent 8x8 map.
Layer/stride/shape/dtype/version metadata is recorded; layer 5 in the no-attention
model is a different stage and must not be substituted silently.

### Experimental training choices

- `frozen_head_pilot`: only BCE trains the projection. Frozen detector parameters
  and BatchNorm state stay unchanged. This is NOT the final joint method.
- `joint`: detector and projection train together. One original frame supplies
  coarse BCE and one uniformly sampled candidate crop from that frame supplies
  the installed Ultralytics detection loss. There is no backpropagation through
  hard selection and no extra inference detections from the coarse pass.
- Actual loss: sum of YOLO loss components divided by batch size, plus
  `lambda_rel * mean_BCE_over_valid_cells`. Fine GT visibility defaults to 0.3.
- Both modes initialize a fresh projection. Joint mode starts from the supplied
  tiled detector; it does not implicitly load a preceding frozen head.
- AdamW, warmup + cosine decay, optional CUDA AMP, gradient clipping, no EMA,
  and early stopping on total validation loss. No mosaic or fine-crop augmentation
  is added in this experimental joint recipe; this is not Ultralytics' default
  detector training recipe. Validation crop sampling stays fixed across epochs.
- `lambda_rel=1`, `pos_weight=1`, threshold 0.5 and coarse size 640 are starting
  configurations, not tuned recommendations or values claimed by the method.
- Resume restores optimizer, scheduler, scaler, RNG, data-loader generator and
  YOLO26 end-to-end loss updates at epoch boundaries. Source/config/checkpoint/
  split fingerprints must match. A changed config requires a separately named run.
- `best.json` points to `best_epochNNNN/{detector.pt,lts.pt,metadata.json}`.
  Always load that exact detector/head pair, including for frozen pilots.
  `resume.pt` is training state, not a fine-detector inference checkpoint.

## Commands

Run from the repository root with the installed `traffic_light` environment.
The new COCO adapter additionally needs `pycocotools` (`pip install -e '.[research]'`
when installing the optional dependencies); it was already installed for these
CPU checks. Keep the verified Ultralytics 8.4.32 version for this implementation.

### 1. Inspect the detector recipe without training

```bash
python scripts/train_detector.py configs/train/detector_kayuan.yaml \
  model.arch=yolo26m model.attention=none tiling.type=uniform \
  output.project=runs/lat_yolo/kayuan \
  output.name=yolo26m_p2_tiled_none 'train.seeds=[0,42,100]' --dry-run
```

After approval, remove `--dry-run` to train. Repeat with `model.attention=cbam`
and `se`, each with a distinct output name. A1 uses `tiling.type=none` and a
distinct untiled run name. The existing trainer prepares tiled train/val data
as needed. Do not overwrite old runs or reuse a tiled-cache suffix with changed
tiling parameters. Use approved split/class-map files for each dataset.

### 2. Frozen pilot, then joint training

Set `DETECTOR` to the corresponding trained tiled detector, not a generic
80-class checkpoint. `dataset.data_yaml` checks class IDs/names against it.

```bash
DETECTOR="runs/lat_yolo/kayuan/yolo26m_p2_tiled_none_seed0/weights/best.pt"
python scripts/train_lts.py configs/train/lts_kayuan.yaml \
  detector.weights="$DETECTOR" features.feature_stage=4 \
  training_mode=frozen_head_pilot train.seed=0 train.device=cpu \
  out=runs/lat_yolo/kayuan/lts_frozen_none_seed0 --dry-run

python scripts/train_lts.py configs/train/lts_kayuan.yaml \
  detector.weights="$DETECTOR" features.feature_stage=4 \
  training_mode=joint train.seed=0 train.device=cpu \
  out=runs/lat_yolo/kayuan/lts_joint_none_seed0 --dry-run
```

These dry runs do not load weights or validate dataset contents. After approval,
use `CUDA_VISIBLE_DEVICES=1 python ... train.device=cuda:0` without `--dry-run`.
For SE/CBAM use the matched detector and `features.feature_stage=5`. The LTS
trainer accepts one seed per invocation; repeat 0, 42, 100 with distinct outputs.
For a bounded smoke run, `--stop-after-epochs 1` preserves the planned schedule;
resume with the exact same arguments plus `--resume <run>/resume.pt`.

### 3. Validation and threshold selection

Use the best matched pair, for example:

```bash
RUN="runs/lat_yolo/kayuan/lts_joint_none_seed0"
PAIR=$(python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p/json.loads((p/"best.json").read_text())["directory"])' "$RUN")

python scripts/benchmark_lts.py configs/eval/lts_kayuan.yaml \
  detector.weights="$PAIR/detector.pt" tile_selector.lts.weights="$PAIR/lts.pt" \
  eval.experiment_id=kayuan_joint_none_val eval.seed=0 --dry-run
```

When ready to measure, remove `--dry-run` and select the inference device. This
evaluates uniform, param-free, and LTS using the SAME updated detector. To add the
independent baselines, set `tile_selector.type=[none,tile_cnn,gridnet,param_free_avg,lts]`
and provide their trained checkpoint paths. Selectors are loaded one at a time.

Validation threshold sweep (execution requires available trained pairs):

```bash
for THR in 0.3 0.5 0.7 0.9; do
  CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_lts.py configs/eval/lts_kayuan.yaml \
    detector.device=cuda:0 detector.weights="$PAIR/detector.pt" \
    tile_selector.lts.weights="$PAIR/lts.pt" \
    tile_selector.lts.threshold="$THR" tile_selector.param_free_avg.threshold="$THR" \
    eval.experiment_id="kayuan_val_thr${THR}" eval.seed=0 \
    eval.out="outputs/lat_yolo/kayuan/val_seed0_thr${THR}.json" || break
done
```

Also run an explicit `tile_selector.lts.force_keep_all=true` control and the
equivalent parameter-free control. Threshold 0 does not substitute for a forced
all-tiles control. Compare threshold curves at matched kept-tile fractions or FPS;
equal numerical thresholds need not have equal calibration. Tune pos_weight,
lambda and stage on validation only; pos_weight needs retraining, threshold does not.

### 4. Test and collect three seeds

Freeze validation choices first. Change `dataset.images/labels` from val to test,
set `eval.seed`, matched weights for that seed, and a unique `eval.out`. Use one
shared `eval.experiment_id`, e.g. `kayuan_joint_none_test`, across the three seeds.
Repeat minimally instrumented timing with `eval.profile_stages=false` under a
separate experiment ID/output. Never aggregate profiled and minimal runs together.

```bash
python scripts/collect_e02.py \
  outputs/lat_yolo/kayuan/test_seed0.json \
  outputs/lat_yolo/kayuan/test_seed42.json \
  outputs/lat_yolo/kayuan/test_seed100.json \
  --out outputs/lat_yolo/kayuan/test_summary.json
```

Produces JSON with numeric mean/sample SD and CSV with `mean +/- SD` cells. This
is variation across training seeds, not a confidence interval or timing-repeat SD.
Weight SHA is allowed to differ across seeds, but must match across selectors
within a seed. The collector also checks dataset, source, class map, settings,
hardware, warmup, metric backend and profiling mode. Use separate experiment IDs
for different attention variants, frozen/joint runs, thresholds, or timing repeats.

For another dataset, override all `dataset.*` paths/name/class map/manifest,
detector and selector checkpoints, and output names. Do not reuse Kayuan class maps
or weights implicitly. No dataset was downloaded, modified or re-split here.

## Timing and metric definitions

- FPS = number of original images / `wall_s`; mean latency = `1000*wall_s/N`.
  `time_min=wall_s/60` is evaluation time, not detector training time.
- `wall_s` includes tiling, coarse pass, selection, crop preparation, fine YOLO,
  remap, merge and per-image loop bookkeeping. Images are preloaded; disk I/O,
  weight loading, warmup, hashing, accuracy evaluation and visualization are excluded.
- `select_s`: complete coarse/selection path (zero for uniform); `detect_s`:
  crop preparation + batched fine YOLO + coordinate remap; `merge_s`: class-aware
  cross-tile NMS. Their sum is <= wall time because outer work also takes time.
- Detailed `stage_latency_ms`: coarse preprocess/backbone, relevance prediction,
  tile scoring/selection, crop preparation, fine detector, remap, NMS and total.
  Fine-detector time includes Ultralytics' internal preprocessing/postprocessing,
  not only a neural forward. Mean/p50/p95 are over original images.
- CUDA is synchronized on the pipeline's devices. Complete-path warmup and one
  resident selector are used. Profiling adds sync/measurement overhead: report
  `profile_stages=false` wall results separately from the breakdown run.
- `fine_forward_batches` counts batches of kept tiles, not original-image FPS.
  `coarse_backbone_calls` counts the shared coarse YOLO calls, not GridNet calls.
- `selector_gt_coverage` counts GT boxes covered by at least the configured fraction
  (default 0.5) by ONE selected tile. It is not cell recall or detector recall.
- COCO bbox AP uses original-image areas, class-aware IoUs 0.50:0.05:0.95 and
  `coco_max_dets=100`; undefined area ranges are null. `max_det=300` in YOLO/merge
  is a separate cap. YOLO26 end-to-end prediction may not use ordinary intra-tile NMS.
- P/R/F1 are project micro matching at IoU 0.5 and the configured detector conf
  (new template 0.001). They are not the best-F1 threshold from `model.val()`.
  Report a separate fixed-confidence operating point when needed.
- Each JSON has resolved YAML, checkpoint/config/source/version/hardware hashes,
  timestamps, manifest and ordered image/label content fingerprints. These runtime
  fingerprints are not a new leakage audit. Optional prediction JSON and images
  show selected tiles (green), GT (white), and the original-coordinate heatmap.
- New outputs are immutable. `--overwrite` from the legacy evaluator does not
  permit overwriting `lat_yolo_v1` results. Pick new output names.

## Remaining experimental decisions

P8 still needs owner-approved trained detector pairs, compute budget and full
training/validation/test runs. Freeze feature stage, lambda, pos_weight, threshold,
recipe length, operating confidence and hardware before final testing. Joint LTS
changes detector weights: compare each selector to uniform on that SAME joint
checkpoint; separately report pre-joint vs post-joint uniform as a training effect.
Do not label frozen pilot rows as A5/A8/A9 joint results.

E02 A1-A9/B1 and E03/E04/E08 now have runnable components, but no measured research
rows. E01 SOTA baselines, E05 other-backbone matched training, E06 cross-dataset
benchmark protocols, and final E07 qualitative figures remain unexecuted. Official
VisDrone ignore-region handling and external benchmark scripts are not added.
GFLOPs for the adaptive path, GPU AMP equivalence, GPU legacy-weight parity and
real GPU timing are not verified. The whole dataset is still preloaded in host RAM,
as in the existing evaluator; plan memory capacity for high-resolution datasets.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

At this implementation checkpoint: **209 passed, 13 skipped** on CPU, including
three synthetic YOLO26m-P2-P5 train/resume/evaluate integrations. The skipped tests
need GPU/assets or explicit full-run enabling; no GPU parity claim follows.
`git diff --check` passed. Long GPU training was not started.
