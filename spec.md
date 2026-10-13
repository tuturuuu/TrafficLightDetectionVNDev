# SPEC.md — LAT-YOLO Research-Aligned Implementation

**Project:** LAT-YOLO: Learnable Adaptive Tiling with Attention for Small Object Detection in Traffic and Aerial Scenes  
**Repository:** `https://github.com/tuturuuu/TrafficLightDetectionVNDev`  
**Target starting point:** branch `refactor`, directory `pipeline/`  
**Companion document:** [`implementation_plan.md`](implementation_plan.md)  
**Status:** Implementation specification. The existing repository does **not** yet implement the paper's feature-guided LTS.  
**Source review:** 2026-10-12. Before editing, inspect the checked-out revision and record its commit SHA; the remote repository may change.

> **MANDATORY FOR CODING AGENTS**
>
> Implement the *method described in the LAT-YOLO manuscript*, not an arbitrary heatmap-based CNN selector. Preserve `GridNet` and `tile_cnn` as independent baselines. Do **not** silently call GridNet “LTS,” reuse GridNet scores as LTS scores, assert unmeasured speedups, or replace the paper's shared-backbone 1×1 projection with an auxiliary CNN. Distinguish `implemented`, `verified`, `experimental`, and `proposed` in all outputs.

## 1. Purpose and scientific question

Detect tiny objects in high-resolution traffic and aerial images without paying the latency of running a full-resolution detector on **every** overlapping tile. The manuscript introduces **Learnable Tile Selection (LTS)** as its methodological contribution within the overall **LAT-YOLO** framework.

The primary question is **not** whether tiling is useful; fixed tiling is already established. Instead:

> Can a task-trained 1×1 projection of the *same YOLO detector's coarse backbone features* select object-relevant tiles more effectively than channel-averaged activations and independent CNN selectors, while preserving most fixed-tiling detection accuracy and reducing measured end-to-end latency?

A valid outcome may be negative. Report any accuracy loss, increased coarse-pass overhead, or lack of speedup; never manufacture a favorable result. Reducing tiles does not by itself prove faster inference.

### In-scope datasets

- **2024kaiyuan Subset 2:** primary traffic/countdown study; confirm dataset variant, classes, video-sequence identities, and image-label mapping.
- **BSTLD:** secondary traffic study.
- **VisDrone2019-DET:** aerial domain.
- **SeaDronesSee:** aerial/maritime domain.

Do not assume dataset availability. `pipeline/README.md` recorded BSTLD and VisDrone as unavailable at the time of its last update. Confirm current files and data licenses before running.

### Non-goals

- Do not replace YOLO with GridNet.
- Do not learn free-form tile coordinates, clustering, reinforcement-learning cropping, segmentation masks, or dynamic overlap; **the manuscript uses a regular candidate-tile grid plus learned *selection*.**
- Do not add an independent feature extractor for LTS; its added relevance head is exactly one learnable 1×1 convolution, followed by sigmoid only at scoring time.
- Do not erase or rename existing GridNet, TileCNN, legacy checkpoints, or historical experiment records.
- Do not claim that CBAM/SE are novel modules; evaluate them as attention variants.
- Do not run a detector's full detection head solely to obtain the coarse features if a backbone-only path is available.

## 2. Verified repository baseline (not an LTS implementation)

Repository files checked at `refactor/pipeline/`:

| Existing component | Path | Observed behavior |
|---|---|---|
| Orchestration | `src/adaptile/pipeline.py` | `Tiler -> Selector -> YOLO Detector -> Merger`; `ImageResult` currently tracks selection and detection timing. |
| GridNet model | `src/adaptile/models/gridnet.py` | Independent four-block CNN on 256×256 image; predicts grid logits through pooling and two 1×1 conv layers. Uses ground-truth **box centers** for training targets. |
| GridNet adapter | `src/adaptile/selectors/gridnet.py` | `gridnet.propose(image)` -> boolean grid -> `grid_mask_for_tiles()`. |
| Naive classifier | `src/adaptile/selectors/tile_cnn.py` | Independent CNN predicts each tile's keep probability. |
| YOLO wrapper | `src/adaptile/detectors.py` | `YOLO(weights).predict(crops)`, maps tile boxes to original image. No exposed coarse feature API yet. |
| Tiling | `src/adaptile/tiling.py` | Candidate tiles `(x1,y1,x2,y2,row,col)` with overlap. `grid_mask_for_tiles()` is **boolean OR**, not float-map max aggregation. |
| Selector registration | `src/adaptile/selectors/__init__.py` | `none`, `gridnet`, `tile_cnn`; no `lts` or `param_free_avg`. |
| Training | `src/adaptile/training.py` | Separate detector and selector trainers; `train_selector` currently accepts only `gridnet` or `tile_cnn`. |
| Config parser | `src/adaptile/config.py` | Config-driven registry plus a selector-type list; add the two new modes without breaking existing configs. |
| Evaluation | `scripts/evaluate.py`, `src/adaptile/metrics.py` | Class-aware metrics, pipeline rows, image-level wall FPS. AP_S and full timing decomposition not present. |
| Attention | `src/adaptile/models/attention.py` | CBAM/SE implementations contain lazy layer creation; audit before training. |
| Architecture | `configs/models/yolo26m-modified.yaml` | Contains P5 and four detection outputs P2/P3/P4/P5, **despite manuscript text about removing the deepest detection stage**. |

**Conclusion:** GridNet is a legitimate independent learned-selection baseline. It is not Equation (7) LTS because it does not consume the selected YOLO backbone feature tensor.

Repository reference links (verify on the actual checkout):

- [pipeline](https://raw.githubusercontent.com/tuturuuu/TrafficLightDetectionVNDev/refactor/pipeline/src/adaptile/pipeline.py)
- [gridnet](https://raw.githubusercontent.com/tuturuuu/TrafficLightDetectionVNDev/refactor/pipeline/src/adaptile/models/gridnet.py)
- [detectors](https://raw.githubusercontent.com/tuturuuu/TrafficLightDetectionVNDev/refactor/pipeline/src/adaptile/detectors.py)
- [training](https://raw.githubusercontent.com/tuturuuu/TrafficLightDetectionVNDev/refactor/pipeline/src/adaptile/training.py)
- [evaluation](https://raw.githubusercontent.com/tuturuuu/TrafficLightDetectionVNDev/refactor/pipeline/scripts/evaluate.py)
- [attention](https://raw.githubusercontent.com/tuturuuu/TrafficLightDetectionVNDev/refactor/pipeline/src/adaptile/models/attention.py)

## 3. Source of truth: manuscript versus engineering choices

The user-provided `LAT_YOLO.pdf` is the method source. Implement these elements as specified:

| Manuscript | Required semantics |
|---|---|
| Section III-A, Figure 1 | Coarse pass -> feature-guided selection -> fine pass -> remap and NMS. |
| Section III-B | Study CBAM and SE under matched backbone/tiling; verify the deepest-stage claim before reporting it. |
| Eq. (7), III-C.2 | `S = sigmoid(Conv1x1(F_c))`: **one output channel**, `C+1` learnable weights and bias. |
| Eq. (8) | `S_avg[i,j] = mean_c(sigmoid(F_c[c,i,j]))`: parameter-free comparator, **sigmoid before channel mean**. |
| Eq. (9), III-C.3 | For tile footprint `Ω_k`, `p_k = max(S[Ω_k])`; select if `p_k > δ`. |
| Eq. (10), III-C.4 | `L = L_det + λ L_rel`; `L_rel` is BCE with targets derived from GT-overlap on the relevance map. |
| Algorithm 1 | Generate regular candidate tiles on the original image, detect selected full-resolution tiles, remap to original coordinates, NMS merge. |
| Table II | Isolate fixed tiling, independent CNN, parameter-free projection, LTS, CBAM, SE on matched YOLO26m. |

**Underspecified in manuscript — agent must document and resolve, never invent as a published fact:** exact YOLO feature stage, downsample/letterbox size, grid size/overlap, fixed `δ`, relevance weight `λ`, class imbalance weighting, `min_keep` fallback, whether GT includes boxes with limited visible fraction, precise joint-training schedule, batching strategy, feature target coordinate mapping, and hardware/profile mode. Choose conservative configuration-driven defaults **for experiments only**, label them as implementation decisions, and record results before updating the paper.

## 4. Required inference architecture

```text
Original BGR image I (H × W × 3)
       │
       ├── create fixed/overlapping candidate tile rectangles on ORIGINAL I
       │
       └── YOLO-compatible coarse preprocessing (resize/letterbox + RGB + normalize)
                    │
                    ▼
           SAME detector's YOLO backbone, run ONCE
                    │  F_c: [B,C,Hf,Wf] from configured feature stage
                    ├────────────┬─────────────────────────┐
                    ▼            ▼                         │
        LTS 1×1 Conv logits   PARAM-FREE AVG                 │
                    │            │                         │
              sigmoid         sigmoid -> mean_c             │
                    └────────────┘                         │
                          S: [B,1,Hf,Wf]                    │
                                │                          │
               map original-image tile footprint to S       │
                                │                          │
                     max over footprint                     │
                                │                          │
                        threshold δ                         │
                                ▼                          │
                    selected original tiles                │
                                │                          │
                  crop + fine YOLO detector ◄───────────────┘
                                │
                    coordinate remap + class-aware NMS
                                │
                         final detections
```

The line connecting backbone to fine detector denotes **shared weights**, not that the coarse feature tensor itself must be forwarded into the fine-stage computation. Fine detection evaluates *selected original-resolution crops* using the same YOLO detector architecture and weights.

### 4.1 Backbone feature extraction contract

- Obtain the coarse tensor from the **same loaded model instance/weights** used for fine detection (or from a strictly verified shared-weight view). Never instantiate an untracked second YOLO or an unrelated encoder under the name `lts`.
- Select a **configurable**, documented backbone stage. For a FPN/skip-connection graph, respect layer dependencies; naive `nn.Sequential(*model[:N])` may be incorrect. Prefer verified graph-aware feature extraction or a carefully managed forward hook, with no accidental full head work.
- Document source model SHA256, feature-layer identifier, stride, tensor shape, preprocessing, dtype, and version. Reject incompatible checkpoints rather than silently loading wrong features.
- Coarse branch in **training** must support gradients when joint training. In **inference**, use eval/no-grad appropriately and avoid retaining activations after each image.
- Stage P2/P3/P4 comparisons, if needed, are **design ablations**; do not silently use concatenation/FPN fusion as it would change the `C+1`-parameter novelty claim.

### 4.2 Learnable projection

Let `F_c` have shape `[B,C,Hf,Wf]`. Implement:

```python
# Illustrative contract, not a drop-in integration with Ultralytics
class LTSHead(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = torch.nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, coarse_features):
        return self.proj(coarse_features)  # raw logits [B,1,Hf,Wf]
```

Training loss: `BCEWithLogitsLoss(logits, targets)` (numerically stable). Inference map: `torch.sigmoid(logits)`. For exact Equation (7) **no extra hidden conv, normalization, attention, or CNN in the projection head**. Assert `sum(p.numel() for p in lts_head.parameters()) == C+1`.

### 4.3 Parameter-free comparator

Compute `S_avg = torch.sigmoid(F_c).mean(dim=1, keepdim=True)`; **not** `torch.sigmoid(F_c.mean(dim=1))`. It shares the same feature extractor, tile geometry and fine detector as LTS. It has zero *additional relevance-head* parameters, not zero total YOLO parameters.

### 4.4 Tile scoring and coordinate systems

Original-space candidate tile: `(x1,y1,x2,y2)` in pixels of the original image, possible overlap. The coarse image might be resized and letterboxed, and `F_c` may have stride. Implement a well-tested mapping:

1. Original pixel coords -> coarse resized/letterboxed coords (exact scale and padding).
2. Coarse coords -> nominal feature-map cell-footprint indices, with clipped bounds and documented interval convention.
3. Aggregate **all overlapping feature-map cells** using `max`. For degenerate subpixel regions, define a deterministic nearest-cell fallback and log it; do not index an empty array.
4. `selected = tile_score > threshold`. Honor a clearly specified `min_keep` policy. The **default paper-faithful setting should be `min_keep=0`**, because Algorithm 1 does not specify an emergency fallback. If an alternative (`min_keep>=1`, top-k) is evaluated, name/report that variant separately.
5. Preserve the candidate tile order and original-image coordinates; do not collapse overlap or change tile shape for selected tiles.

**Critical distinction:** Existing `grid_mask_for_tiles()` ORs booleans from an 8×8 GridNet mask. LTS needs a **float map** and a **max score** mapped from a chosen YOLO feature-map size. Implement a new reusable scorer rather than misusing the boolean utility.

### 4.5 Fine pass / merging

- Crop only selected tiles from the original full-resolution image; use exactly the same per-tile preprocessing, fine detector weights, confidence threshold, batch size, border padding policy and remapping logic as fixed tiling.
- Merge boxes using the same class-aware NMS settings for all comparable runs.
- Handle zero selected tiles safely: return `[]` without calling `YOLO.predict([])`.
- The coarse pass is for *tile relevance*, **not an additional source of final bounding boxes** unless a new separately labeled method is being evaluated.

## 5. Supervision and training contract

### 5.1 Relevance targets

From original GT boxes, generate a map `[B,1,Hf,Wf]` aligned to the actual coarse feature geometry. A feature location is positive if its **defined spatial footprint overlaps any GT bounding box**. This is different from GridNet's center-cell label. Document whether boxes are clipped, filtering classes and visibility, and the exact feature-cell footprint approximation used (do not confuse nominal cell coverage with the full neural receptive field).

- Use appropriate transformations when coarse preprocessing letterboxes the image.
- Empty-object images produce all-zero targets and must not yield NaN loss.
- Maintain channel and batch dimensions explicitly; validate multi-object and boundary cases.
- BCE positive weighting is a tunable training hyperparameter **not specified numerically by the paper**. Document it and check class imbalance; it is not a new architecture contribution.
- Train relevance predictor exclusively with labels from the training partition. Validation is for model selection and threshold tuning; test is held out.

### 5.2 Required two training modes, reported accurately

**Mode 1 — head-only sanity check (engineering pilot; NOT paper joint training):** load a trained detector, freeze backbone, train only `LTSHead` with relevance BCE. This confirms the geometry, target generation and informativeness of the chosen feature stage. Save such checkpoints as `lts_frozen_*`; do not present this run as if it implements Equation (10) fully.

**Mode 2 — paper-aligned joint optimization (required for final claim):** in one training workflow, optimize `L_det + λ*L_rel` with a shared detector backbone. `L_det` comes from the detector's actual YOLO training objective; `L_rel` from the coarse feature map and GT-aligned relevance map. Verify that the shared backbone receives both losses where intended, and the LTS projection receives `L_rel` gradients. A hard `score > δ` decision is non-differentiable: **do not claim detection loss backpropagates through hard tile selection**. Detection training may use standard train/tiled crops while the relevance branch learns from corresponding coarse full images; implement explicit aligned sampling and explain the procedure.

Use the installed Ultralytics version's supported loss/trainer APIs, confirmed by smoke tests. Do not fabricate a custom `model.loss()` integration without checking expected batch formats, outputs, callbacks, optimizer parameter registration and checkpoint restore behavior. Prefer a well-scoped trainer subclass or explicit training module with integration tests.

Save in the LTS checkpoint: head weights, model/feature metadata, training mode, dataset split identifier, backbone/checkpoint fingerprint, config/hyperparameters (`λ`, loss/pos_weight, image size, seed), software versions. For joint training, persist the updated shared detector checkpoint as well; fine inference must load its **matched** detector weights.

### 5.3 Attention variants

SE channel attention and CBAM channel-plus-spatial attention are **fine-detector architecture options**, not LTS substitutes. Run matched `YOLO26m` backbone variants (`none`, `CBAM`, `SE`) and state *precisely* which layers contain modules. If replacing P5 or altering detection scales, record a distinct architecture ID and retrain affected baselines. Before training check modules are on the right device and have registered parameters **before** the optimizer is built; existing `attention.py` uses lazy module creation and needs audit.

## 6. Config/API requirements (proposed additions)

Extend existing config-driven registry, maintaining compatibility with `none`, `tile_cnn`, `gridnet`. Suggested selector names: `param_free_avg`, `lts`. Expose a shared `CoarseFeatureProvider` that receives a reference to the detector, not another independently constructed model.

**Example *proposed* YAML schema; implement/update parser accordingly — not a claim this config runs today:**

```yaml
dataset:
  name: kayuan_subset2
  images: data/kayuan_clean/test/images
  labels: data/kayuan_clean/test/labels
tiling:
  type: uniform
  tile_size: 640
  overlap: 0.20
  pad: false
tile_selector:
  type: [none, gridnet, tile_cnn, param_free_avg, lts]
  # pre-existing classifier configurations intentionally retained
  gridnet: {weights: weights/gridnet_clean.pth, threshold: 0.10}
  tile_cnn: {weights: weights/tile_cnn_clean.pth, threshold: 0.50}
  param_free_avg:
    feature_stage: CONFIGURED_BACKBONE_STAGE
    coarse_imgsz: 640
    threshold: 0.50
    min_keep: 0
  lts:
    weights: weights/lts_joint_clean.pt
    feature_stage: CONFIGURED_BACKBONE_STAGE
    coarse_imgsz: 640
    threshold: 0.50
    min_keep: 0
detector:
  type: yolo
  weights: runs/yolo26m_tiled_clean/weights/best.pt
  conf: 0.25
  imgsz: 640
  batch: 16
  device: 0
merger:
  type: nms
  iou_thr: 0.50
  max_det: 300
eval:
  metrics: [map50, map50_95, precision, recall, fps]
  n_warmup: 3
  out: outputs/kayuan_clean/e02_selector_comparison.json
```

This example intentionally uses placeholder paths and `CONFIGURED_BACKBONE_STAGE`. Agent must not pretend referenced checkpoints/datasets exist or invent an actual stage. The existing loader builds selectors before binding them to the detector; revise construction via `from_config(..., detector=...)`, `bind_detector()`, or similar explicit, tested dependency injection. Avoid global mutable model singletons.

Suggested typed interfaces:

```python
@dataclass
class FeatureBatch:
    features: torch.Tensor          # [B,C,Hf,Wf]
    coarse_geometry: object         # actual resize/letterbox/stride mapping

@dataclass
class SelectionResult:
    keep: np.ndarray               # [N] bool, input tile order
    scores: np.ndarray             # [N] float, diagnostic/reproducible
    relevance_map: np.ndarray | None
    timings_ms: dict[str, float]   # preprocess/backbone/score/selection
```

Keep the public simple selector contract `select(image, tiles) -> np.ndarray[bool]` working for legacy modes; a richer result can be an opt-in `select_with_details()` and pipeline adapter. No breaking changes to existing external callers without migrations and tests.

## 7. E02: mandatory ablation matrix

All rows require the same dataset split and metric evaluator. Within each *tiling-selector* comparison, use the **same tiled detector checkpoint**. Do not conflate a backbone change or attention change with a selector effect.

| ID | Paper configuration | Tiling | Selection | Attention | Status at reviewed revision |
|---|---|---|---|---|---|
| A1 | Baseline | none | none | none | Existing pipeline can run; confirm checkpoint and clean split. |
| A2 | + Fixed tiling | fixed | all | none | Existing pipeline can run; confirm tiled detector checkpoint. |
| A3 | + Naive filter | fixed | independent per-tile CNN | none | `tile_cnn` exists; evaluate with matched no-attention tiled detector. |
| A4 | + Param-free avg. | grid candidates | Eq. (8) | none | **Implement.** |
| A5 | + LTS (learned) | grid candidates | Eq. (7) | none | **Implement and train.** |
| A6 | + Fixed + CBAM | fixed | all | CBAM | Need YOLO26m+CBAM matched config/checkpoint. |
| A7 | + Fixed + SE | fixed | all | SE | YOLO26m+SE config exists; validate implementation/checkpoint. |
| A8 | + LTS + CBAM | grid candidates | Eq. (7) | CBAM | **Implement/train matched LTS.** |
| A9 | + LTS + SE (LAT-YOLO) | grid candidates | Eq. (7) | SE | **Implement/train matched LTS.** |
| B1 | + Independent GridNet (**additional baseline**) | grid candidates | independent GridNet CNN | none | Existing `gridnet`; preserve and evaluate. |

The A1 detector may be trained on full images, whereas A2–A5/B1 use a tiled detector; state this difference. For A6 versus A8 and A7 versus A9, use *matched* attention detectors. If historic results used `CBAM+SE` jointly, label them separately; **do not insert them into CBAM-only or SE-only rows**.

Metrics for Table II: `mAP50`, `mAP50-95`, `Recall`, `FPS` for 2024kaiyuan Subset 2 and BSTLD. Retain precision, F1, average tiles kept, and timestamps in raw outputs. Compare parameter-free vs LTS not only at the same numerical threshold, but also at matched keep fraction/FPS or along a threshold sweep if score calibration differs.

## 8. Other manuscript experiments

| ID | Paper element | Deliverable |
|---|---|---|
| E01 | Table I (2024kaiyuan Subset 2) | LAT-YOLO row: mAP50, mAP50-95, precision, recall, FPS using final A9 and matched SOTA protocol. |
| E03 | Table III | Thresholds `δ in {0.3,0.5,0.7,0.9}` + all-tile control: tiles processed %, mAP50, recall, FPS; reuse weights. |
| E04 | Table IV | Fixed/LTS × CBAM/SE: mAP50, FPS, attention parameter deltas. Reuse A6–A9 when protocol matches. |
| E05 | Table V | YOLOv8m, YOLO11m, YOLO26m with matched baseline vs LAT; log backbone-specific feature stage/checkpoint. |
| E06 | Table VI | VisDrone and SeaDronesSee: baseline, fixed, LAT; mAP50, mAP50-95, `AP_S`, recall, FPS. |
| E07 | Figure 2 | GT / untiled baseline / fixed / LAT / relevance-map visualizations on Kayuan and VisDrone; selected tile boundaries recommended. |
| E08 | Section V-F | Coarse preprocess + backbone, relevance prediction + tile selection, fine tile infer, NMS/merge, **total end-to-end latency**; fixed-tiling comparator. |

Never fabricate missing results; the manuscript's `[0.97]`, `[]`, `[value]`, and unfilled figure are **placeholders**, not verified findings.

## 9. Experimental rigor and data integrity

The repository `README.md` explicitly warns that historic Kayuan train/val/test were randomly split **by frame** from shared video sequences and that old evaluation was class-agnostic. Consequently:

1. Create and version **video/group-disjoint** train/validation/test manifests for Kayuan. Audit SeaDronesSee grouping and near-duplicates; document when group identifiers do not exist rather than assuming independence.
2. Train detector and selectors **only** on training groups. Choose threshold, λ, feature stage, early stopping, and configurations on validation. Evaluate held-out test only after choices are fixed.
3. Do not assume old weights are valid on the new test partition; audit which source images/sequence groups each checkpoint saw. Retrain when contamination cannot be excluded.
4. Standardize class-aware IoU matching, per-class AP, COCO-style AP/`AP_S` on UAV datasets as appropriate. Verify evaluator against a trusted reference using identical saved predictions.
5. Record dataset provenance and annotation versions, class-map IDs, split manifest SHA256, seed, config, model checkpoint hashes, code commit, torch/Ultralytics/CUDA, GPU and AMP mode, batch size, input resolution, NMS parameters and warmup.
6. Do **not** compare mAP or FPS as matched evidence across different tile sizes, overlap, input sizes, detector checkpoint/training data, hardware, device, batch settings or evaluator implementations without explicitly labeling the differences.

### Speed measurement definition

- Report wall-clock inference per **original image**, including coarse preprocessing/backbone, relevance map computation, grid scoring, full-resolution crop preparation, fine model inference, remapping and NMS.
- Exclude image disk I/O only if excluded consistently for every method and stated in paper.
- Warm up **the complete execution path**, not merely a subset of two tiles when measuring full-tiling throughput; synchronize CUDA around timed regions; track latency distribution (mean, median, p95) and FPS.
- Report fine-forward count and selected tile ratio; `1 - K/N` is **tile reduction**, not automatically GPU speedup.
- Profile the overhead break-even regime. Compare to fixed tiling and untiled YOLO separately; any advantage over fixed does not imply advantage over untiled.
- Keep selected/total tile definitions consistent across overlapping candidate grids and GridNet's *different* 8×8 selection-map grid.

## 10. Acceptance criteria / falsifiable invariants

Implementation may be declared **research-ready** only if:

- **Feature identity:** For a chosen YOLO checkpoint, coarse extractor output equals a verified capture from the configured layer in that **same** network under identical preprocessing/eval mode (within numeric tolerance).
- **Parameter count:** LTS head alone has `C+1` parameters; param-free head has zero.
- **Equation parity:** Eq. (7) and Eq. (8) tested on deterministic synthetic tensors, including noncommutativity of sigmoid and mean.
- **Geometry:** Correct GT->feature and tile->feature mapping for landscape/portrait, non-square images, letterbox, odd edge sizes, overlapping tiles, multi-object labels and empty images.
- **Gradient test:** BCE updates LTS head; joint mode demonstrates expected gradient flow to shared backbone and to detector loss parameters; disabled/frozen parameters are reported rather than assumed.
- **All-tiles equivalence:** With forced `keep[:] = True`, same YOLO weights, preprocessing and NMS, predictions equal fixed tiling to tolerance. The LTS coarse pass may still add latency; never assert end-to-end time equality.
- **Zero-tiles safety:** `min_keep=0` empty selection yields an empty detection list and valid timing/metrics.
- **No legacy regressions:** `none`, `tile_cnn`, `gridnet` still load saved weights and pass parity/unit tests.
- **Fair profiling:** Each timing component is measured and sums within reasonable instrument overhead to measured end-to-end time; CUDA timing semantics documented.
- **No leakage:** Train/val/test manifests pass group and duplicate checks, or any exception is explicitly disclosed and final publication is blocked until resolved.
- **Reproducibility:** Running the same config/seed/checkpoints produces matching detections and metrics within documented tolerances; output configs are immutable and stored with results.

## 11. Agent behavior and handoff

When coding, follow [`implementation_plan.md`](implementation_plan.md). Before edits, read `pipeline/README.md`, `RUNBOOK.md`, `HANDOFF.md`, relevant source files, and the manuscript. Work in an isolated branch; preserve old experiments and weights. Make small testable commits **only if explicitly authorized**. Ask for missing data/checkpoint/architecture choices instead of inventing them. Never run long GPU training or replace historical artifacts without owner approval. At every milestone, provide:

- files changed and why;
- implementation mode (`pilot/frozen` vs `joint/paper-aligned`);
- tests run and actual pass/fail results;
- exact commands/configs to reproduce;
- unresolved decisions and deviations from paper;
- table rows that are now measurable, **without** fabricated results.

**Source mapping:** manuscript `LAT_YOLO.pdf`, Sections III–V, Algorithm 1, Tables I–VI; `refactor/pipeline` README, RUNBOOK, HANDOFF, and code links in Section 2. This specification is an implementation contract and includes explicitly marked proposed engineering decisions beyond the paper.
