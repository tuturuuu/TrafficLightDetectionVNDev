# IMPLEMENTATION_PLAN.md — From GridNet Baselines to Paper-Aligned LAT-YOLO LTS

**Target repository:** `https://github.com/tuturuuu/TrafficLightDetectionVNDev`  
**Working directory:** `pipeline/` on branch `refactor` (create a **new feature branch** before changes)  
**Read first:** [`spec.md`](spec.md), `LAT_YOLO.pdf` Sections III–V, repo `README.md`, `RUNBOOK.md`, `HANDOFF.md`  
**Goal:** Implement **Equation (7) feature-guided LTS**, its **Equation (8) parameter-free comparator**, accurate training/evaluation, and reproduce the E02 ablation without discarding valid GridNet/TileCNN work.

## 0. Strict working rules for the agent

1. **Do not replace or relabel GridNet.** It is an independent-image-CNN selector and remains an experiment baseline. Its two 1×1 convolutions do **not** make it the LTS method in the manuscript.
2. **Do not invent unprovided results, dataset availability, checkpoints, hyperparameter values or architecture decisions.** Store proposed defaults in versioned configs marked `experimental`; do not promote them to manuscript facts without measurements.
3. **No destructive file operations.** Do not overwrite checkpoints, datasets, historic `outputs/`, `runs/`, legacy files, or existing configs. Add versioned names and keep historical IDs.
4. **No long GPU jobs without approval.** First implement code, run CPU unit tests, tiny GPU smoke tests where available, then provide reproducible commands for the user to authorize final training/evaluation.
5. **No paper claims before evidence.** In particular do not reuse GridNet numbers as LAT-YOLO numbers; do not claim improved accuracy or faster end-to-end inference without matched measurements.
6. **Use the actual checked-out repository state.** This document is based on the public `refactor/pipeline` source inspected on 2026-10-12. If paths/architecture changed, show a diff of findings and adjust the plan explicitly.
7. **Respect project Git preference:** work on a new branch, show changes and testing first; do not commit, push, merge, or create PR without user authorization.

## 1. What can be retained (do not retrain automatically)

| Historic work | Scientific role | Reuse conditions |
|---|---|---|
| Full-image YOLO | A1 baseline | Correct backbone, clean split, class-aware evaluation. |
| YOLO+CBAM/SE | attention baseline(s) | Same architecture and verified registered parameters. `CBAM+SE` joint model is **not** CBAM-only or SE-only. |
| Full/fixed tiling + YOLO | A2/A6/A7, according to detector weights | Matched tile size, overlap, preprocessing, checkpoint, NMS. |
| Tile + Proposal CNN | A3 Naive Filter | Use existing `tile_cnn` with same fixed/tiled YOLO as other no-attention selection rows. |
| GridNet + YOLO | **B1 additional independent learned-selection baseline** | Preserve model, test with matched detector and clean data. Not A5. |

**Reason previous metrics may require reevaluation:** Repo README documents (a) Kayuan video-frame leakage across train/val/test and (b) class-agnostic legacy evaluation. This is independent of whether GridNet was a scientifically sound design. Retrain only when the detector/selector checkpoint saw data now in the held-out test groups or when architecture/training changed; otherwise reevaluate with the corrected protocol. Preserve a provenance ledger of every decision.

## 2. Plan at a glance

| Phase | Priority | Outcome | Gate before next phase |
|---|---|---|---|
| P0 | Critical | Freeze code/experiment provenance and identify unresolved design decisions | User-approved protocol manifest / known blockers |
| P1 | Critical | Clean datasets, splits, model/metric audits | Group-disjoint checks and trustworthy evaluation |
| P2 | Critical | Reusable, correct YOLO coarse feature extractor | Feature identity and graph-aware execution test |
| P3 | Critical | Relevance labels + learned / parameter-free heads + tile score mapping | Unit tests for equations and geometry |
| P4 | Critical | Integrate selectors into current pipeline, preserve old modes | All-tiles parity and timing tests |
| P5 | Critical | Frozen-feature pilot, then paper-aligned `L_det + λL_rel` training | Gradient-flow and reload smoke tests |
| P6 | High | Matched YOLO26m attention/backbone variants | Parameter registration and architecture parity |
| P7 | Critical | Fair E02 evaluator and E08 profiling | All rows reproducible on small subset |
| P8 | High | Execute final E02/E01/E03/E04/E05/E06/E07/E08 runs | Evidence archive; no placeholders reported as results |
| P9 | High | Documentation, audit and paper-to-code consistency | Final audit checklist signed off |

Phase labels are logical, not time estimates. Stages may be parallelized only if interfaces are stable.

---

## P0 — Create safe feature branch and evidence inventory

**Actions**

- [ ] Check the exact repo root (`git rev-parse --show-toplevel`), branch, SHA, diff, untracked files and available datasets/checkpoints.
- [ ] Branch from `refactor` **without discarding local changes**:

  ```bash
  git status --short
  git fetch origin
  git switch refactor
  git switch -c feature/lat-yolo-lts
  ```

  If local changes or a different branch exist, **stop and ask** rather than `reset --hard` or force-switching.
- [ ] Record baseline outputs, configs, seed, dataset split, timestamps, GPU/hardware, detector `.pt` and selector `.pth` SHA256, model class counts.
- [ ] Verify actual support for each YOLO architecture/attention combination in the installed Ultralytics version.
- [ ] Read the actual manuscript and list its **unknown implementation parameters**: feature stage, coarse resolution/preprocessing, grid overlap and tile size, `λ`, default `δ`, pos_weight, exact training schedule, feature/GT mapping and safety fallback.
- [ ] Save a machine-readable `outputs/audits/baseline_inventory.json` and `outputs/audits/method_decisions.md` (or equivalent), with explicit provenance and status. Do not modify legacy experiment records.

**Files to inspect:** `README.md`, `RUNBOOK.md`, `HANDOFF.md`, `configs/{train,eval,models}/`, `src/adaptile/models/gridnet.py`, `weights/README.md`, `outputs/README.md`.

**Gate:** No uncertain setting is silently hard-coded. Any blocker that affects scientific validity is elevated to the user.

## P1 — Correct evaluation integrity before claiming results

### P1.1 Dataset leakage audit

**Likely additions:** `scripts/data/audit_splits.py`, `scripts/data/group_split_kayuan.py`, `test/test_data_integrity.py`, a versioned split manifest under a **new** dataset revision path.

- [ ] Inspect actual Kayuan filename/video metadata and frame order; reconstruct video/sequence group IDs where possible.
- [ ] Produce train/val/test groups with no video overlap. Check duplicate and near-duplicate samples. Keep a manifest with group IDs, count, seed, original source and split SHA256.
- [ ] Audit SeaDronesSee for comparable overlap; if grouping evidence is incomplete, mark it as a limitation.
- [ ] Do not change existing `data/kayuan/` in place. Use a clearly versioned location such as `data/kayuan_group_v1/` (name is proposed; adapt to local layout).
- [ ] Confirm class mappings, image/label alignment, box coordinate validity and empty images.
- [ ] Prohibit training/fitting thresholds on test data. The historic phrase “trained on val to choose threshold” must be investigated: actual **weight training on val** invalidates val for unbiased model selection unless a second held-out tuning set exists.

**Acceptance:** train/val/test group-ID intersections are empty, with a machine-verifiable report; any exceptions explicitly block final results.

### P1.2 Metric and checkpoint audit

**Inspect/update:** `src/adaptile/metrics.py`, `scripts/evaluate.py`; new regression tests; optionally `src/adaptile/metrics_coco.py`.

- [ ] Preserve the existing class-aware metric pathway and keep class-agnostic historical metrics visibly separated.
- [ ] Test against trusted Ultralytics or COCO evaluator from **identical exported detections**, especially mAP50-95 and `AP_S` for aerial data.
- [ ] Check the correct coordinate system for small-object area: COCO AP_S convention is original-image GT area `< 32²` pixels² (subject to the dataset's official protocol). Do not compute it from resized detector tiles.
- [ ] Audit NMS and confidence filtering for YOLO26's `end2end` mode and any final inter-tile merge; do not confuse model-internal NMS behavior with cross-tile NMS.
- [ ] Identify which old checkpoints are eligible for clean-split reevaluation and which must be retrained due to contaminated training groups.

**Acceptance:** deterministic metric regression tests pass, and the checkpoint provenance report identifies each model's split and architecture.

---

## P2 — Implement shared YOLO backbone feature extraction

**Files:**

- **NEW** `src/adaptile/models/coarse_features.py` (or `src/adaptile/features.py` if better fits code layout).
- **EDIT** `src/adaptile/detectors.py` (expose **the same model instance** to coarse extractor and fine detector).
- **NEW** `test/test_coarse_features.py`.

**Required API**

```python
provider = YOLOCoarseFeatureProvider(
    detector=existing_yolo_detector,
    feature_stage=cfg.feature_stage,
    coarse_imgsz=cfg.coarse_imgsz,
)
feature_batch = provider(image_or_batch, grad_enabled=False)
assert feature_batch.features.ndim == 4
```

**Implementation details**

- [ ] Determine whether `ultralytics.YOLO(weights).model` exposes a safe internal graph traversal / hook at the chosen stage. Preserve `from` edges and saved-activation references. Beware a naive `nn.Sequential` slice.
- [ ] Select one **documented backbone feature tensor** per configuration; verify resolution, stride and channels. Avoid adding FPN fusion/projection layers in this first method (changes the `C+1` novelty claim).
- [ ] Use YOLO-compatible RGB conversion, normalization, letterbox/resize, device/dtype behavior and AMP. Return the exact coarse geometry metadata necessary for coordinate alignment.
- [ ] Run backbone-only compute. If an initial hook prototype still executes the full model head, make that an explicitly marked prototype; optimize to backbone-only **before final speed benchmarks**.
- [ ] Share weights with fine-stage detector and verify identity by comparing the feature tensor against a reference capture from that detector under identical preprocessing.
- [ ] Support `eval/no_grad` and `train/grad` modes; do not retain stale activation tensors between images.
- [ ] Log the stage ID, model architecture fingerprint, input/output shapes, coarse operation timings and device.

**Tests:** same weights -> identical feature output; non-square image; CPU and CUDA where available; forward multiple images without stale-state leakage; model load and restore; attention variant path; batch-vs-single output numerical tolerance.

**Gate:** A pure coarse path and verified feature tensor exist, independent of LTS head.

## P3 — Implement Equations (7), (8), (9) and targets

**Files:**

- **NEW** `src/adaptile/models/lts.py`.
- **NEW** `src/adaptile/relevance_targets.py` (the repository already has `src/adaptile/data.py`).
- **NEW** `src/adaptile/selectors/relevance_utils.py` (shared geometry and float-map tile scoring).
- **NEW** `test/test_lts_math.py`, `test/test_relevance_geometry.py`.

### P3.1 LTS head (Equation 7)

- [ ] `nn.Conv2d(C, 1, kernel_size=1, bias=True)` only.
- [ ] Return **logits** during training; use `torch.sigmoid` for inference.
- [ ] Assert `head_parameters == C + 1`.
- [ ] Confirm output `[B,1,Hf,Wf]`; no additional learnable CNN block.

### P3.2 Parameter-free map (Equation 8)

- [ ] `torch.sigmoid(features).mean(dim=1, keepdim=True)`. Use the same provider.
- [ ] Unit test that `mean(sigmoid(F))` is **not** replaced by `sigmoid(mean(F))`.
- [ ] Confirm no new learnable relevance-head parameters.

### P3.3 GT heatmap / BCE targets (Equation 10)

- [ ] Generate positive cells from **GT overlap** with each cell's nominal spatial footprint in the coarse/feature-map grid (not GridNet center-cell labels).
- [ ] Convert original GT through exact letterbox/resize mapping, clip to valid input and feature bounds.
- [ ] Test GT on grid boundaries, subpixel/tiny boxes, padded regions, multiple objects, empty images, landscape and portrait inputs.
- [ ] Explicitly choose/document how to treat cells representing padding; ignore or mask out those cells in relevance loss unless method justification says otherwise.
- [ ] Support class-agnostic “presence of any target class” supervision by default; document class filters if used.

### P3.4 Float relevance map -> tile scores (Equation 9)

- [ ] `score(tile) = maximum relevance across all intersecting feature cells`.
- [ ] Implement conversion original image pixels -> coarse pixels -> feature cells, with clamp and deterministic handling of empty intersection.
- [ ] Apply strict `>` threshold, not `>=`, unless the method is explicitly revised.
- [ ] Default to paper-aligned `min_keep: 0`; support optional safety fallback as a separately named experiment variation.
- [ ] Keep `grid_mask_for_tiles()` intact for GridNet; do **not** retrofit the LTS mechanism by OR-ing thresholded cells if this changes the score definition or threshold semantics.

**Acceptance tests:** synthetic maps with expected scores, tiny objects across borders, `δ=0`/`δ=1`, float extremes, no NaNs, exact original tile ordering.

---

## P4 — Add selectors and integrate them without breaking old paths

**Files:**

- **NEW** `src/adaptile/selectors/lts.py`.
- **NEW** `src/adaptile/selectors/param_free_avg.py`.
- **EDIT** `src/adaptile/selectors/__init__.py`.
- **EDIT** `src/adaptile/selectors/base.py` only as needed (add optional richer result; preserve existing `.select`).
- **EDIT** `src/adaptile/pipeline.py`, `src/adaptile/config.py`, `src/adaptile/detectors.py`.
- **EDIT** `scripts/evaluate.py` for new modes/selector binding.
- **NEW** `test/test_lts_pipeline.py`, `test/test_selector_compat.py`.

**Do:**

- [ ] Register `lts` and `param_free_avg` in the existing registry.
- [ ] Bind the selector to **the already-loaded detector instance**. Current `scripts/evaluate.py` builds `detector` before `selectors`; leverage this order, but do not build another YOLO inside each selector.
- [ ] Preserve `gridnet`, `tile_cnn`, `none` semantics, saved model loaders and config syntax.
- [ ] Add optional `SelectionResult(scores, keep, relevance_map, timings)` without breaking existing callers expecting `np.ndarray[bool]`.
- [ ] Preserve candidate tiles and final detector call / box remapping / NMS exactly for comparable runs.
- [ ] Safely handle empty kept tile list (`min_keep=0`).
- [ ] Warm up **full coarse + selection + representative full fine path**, including all tiles for fixed mode; do not base timing on only two tiles.
- [ ] Make cross-tile NMS consistent and explicit for all methods.
- [ ] Add a debug option to force keep-all for parity testing.

**Gate:** Forced all-selected LTS yields the same detections as uniform/fixed tiling with the same fine detector, even though LTS incurs extra coarse-pass time. Existing selector parity tests still pass.

## P5 — Train LTS correctly (pilot versus paper-grade)

**Files:**

- **NEW** `src/adaptile/training_lts.py`.
- **NEW** `scripts/train_lts.py` (or extend current train entry point only if interface remains clear).
- **NEW** `configs/train/lts_kayuan.yaml`, plus relevant dataset configs.
- **NEW** `test/test_lts_training.py`, `test/test_lts_checkpoint.py`.
- **EDIT** `src/adaptile/training.py` only for shared training utilities or dispatch, not by silently substituting old GridNet trainer.

### P5.1 Fast sanity-check mode (frozen)

- [ ] Freeze YOLO backbone/attention/detector; train only the 1×1 LTS head on training images' GT overlap maps.
- [ ] Use `BCEWithLogitsLoss`; log positive fraction, BCE, grid/cell recall, selected tile fraction on held-out validation.
- [ ] Save `training_mode: frozen_head_pilot`; do not promote these results as implementation of joint Equation (10).
- [ ] Plot relevance maps and GT overlays from validation. Confirm sensible localization before joint training.

### P5.2 Required final training mode (joint)

- [ ] Implement a real optimization step with `L_total = L_det + λ L_rel`, **sharing detector backbone parameters**.
- [ ] Define sampling: original training image for coarse relevance and corresponding full-resolution tile(s) or tiled samples for YOLO loss; preserve label/coordinate correctness.
- [ ] Ensure optimizer includes all trainable detector, attention and LTS weights **before the first update**, and that gradients reach intended modules. Verify no double optimizer step or inadvertent `detach()`.
- [ ] Hard thresholding is non-differentiable. State that LTS receives supervised relevance gradient, not gradients through hard tile choice.
- [ ] Handle image-level supervision on full frames alongside detection loss on selected or training tiles without leaking validation/test GT at inference.
- [ ] Implement warmup, AMP, reproducibility, checkpoint/resume, and validation using the correct trained/shared detector pair.
- [ ] If the YOLO training/loss interface for the checked-out version makes a correct joint loss infeasible, **stop and report a blocker**; do not silently finish with a frozen head and call it joint training. Propose an accurately labeled two-stage variant only after user agreement.
- [ ] Tune `λ`, positive weighting and training hyperparameters on validation, not test. Avoid claiming manuscript placeholder values were experimentally established.

**Joint-training verification example:** after backward, inspect nonzero gradient norms for LTS projection and the intended YOLO backbone; also verify that the detection head receives `L_det` gradients. When testing loss contributions, isolate each loss and compare gradients with appropriate zeroing.

**Checkpoint requirements:** LTS head state, matched trained detector state/checkpoint SHA, feature-stage metadata, dataset split ID, preprocessing, λ, seed, optimizer/scheduler for resumption, Ultralytics and PyTorch versions, training mode. Refuse incompatible selector-detector combinations.

**Gate:** Full checkpoint loads into one coherent detector+selector pair, both losses run, gradient tests pass, evaluation can produce valid predictions on a tiny disjoint validation sample.

---

## P6 — Match backbone and attention configurations in E02

**Files:**

- **AUDIT / EDIT** `configs/models/yolo26m-modified.yaml`, `src/adaptile/models/attention.py`, `src/adaptile/training.py`.
- **NEW** as needed: versioned architecture files `configs/models/yolo26m_cbam_<version>.yaml` and matched `*_se_*` config; use final names only after reviewing actual architecture.
- **NEW** `test/test_attention_registration.py`, `test/test_yolo_architectures.py`.

- [ ] Set a single **primary YOLO26m reference architecture** for E02; keep attention placements consistent.
- [ ] Resolve manuscript conflict: the text says deepest stage removed, while current `yolo26m-modified.yaml` still has **P5/32** and `Detect(P2,P3,P4,P5)`. Choose to actually modify the network and retrain matched models, or correct the manuscript; **never hide the mismatch**.
- [ ] Add YOLO26m+CBAM if missing; confirm YOLO26m+SE and no-attention variants are structurally comparable apart from intended attention.
- [ ] Audit lazy creation of SE and CBAM parameters. Instantiate/trainable-register them **before optimizer creation**, on correct device/dtype. Add tests for optimizer membership, gradients, save/load and exact parameter counts. Avoid untested architectural edits to pretrained checkpoints.
- [ ] Verify custom modules, names/class mappings and inference compatibility with the installed Ultralytics version.
- [ ] Label old YOLOv8+CBAM / YOLO26m+SE results as cross-model comparisons, not an isolated CBAM-vs-SE ablation.

**Gate:** Training and evaluation smoke tests for same-backbone attention variants pass and there is a frozen architecture manifest/hash per run.

## P7 — Full metrics / profiling / experiment harness

**Files:**

- **EDIT** `src/adaptile/pipeline.py`, `scripts/evaluate.py`, `src/adaptile/config.py`, `src/adaptile/metrics.py`.
- **NEW** `src/adaptile/profiling.py` (if helpful), `scripts/benchmark_lts.py`, `scripts/collect_e02.py`.
- **NEW** `test/test_profiling.py`, `test/test_e02_matrix.py`.
- **NEW** example eval configurations under `configs/eval/lts_*.yaml`.

Implement structured per-image/per-stage timings:

```text
coarse_preprocess_ms
coarse_backbone_ms
relevance_predict_ms
candidate_tile_scoring_ms
fine_crop_prepare_ms
fine_detector_ms
remap_ms
nms_merge_ms
end_to_end_ms
n_candidate_tiles
n_selected_tiles
selector_gt_coverage / tile-selection recall
```

- [ ] Respect GPU async timing and synchronize before and after relevant sections, recording CPU-side end-to-end wall time distinctly if desired. Note the cost of instrumentation and run a minimally instrumented benchmark for FPS.
- [ ] Ensure total reported time includes **all** meaningful inference work, not just `model.forward()`.
- [ ] Exclude disk image loading consistently for all methods; state this in metadata. Warm up all workflows, including full fixed tiling.
- [ ] Report FPS, latency median/p95, mAP50, mAP50-95, precision, recall, F1, average and % kept tiles; use `AP_S` on aerial data.
- [ ] For selector GT coverage, define object as covered if an eligible selected tile sufficiently contains its GT object; state coverage/visibility criterion, distinct from cell recall.
- [ ] Export resolved YAML plus machine-readable JSON/CSV rows, including checkpoint hashes, split manifest SHA, code commit, hardware, batch and thresholds.
- [ ] Support eval runs for `none`, `tile_cnn`, `gridnet`, `param_free_avg`, `lts` on the **same** image list in stable order and the same detector wherever isolation requires it.
- [ ] Add reproducibility tests: forced all-selected parity and serialized predictions comparison.

**Gate:** A tiny smoke-run produces scientifically interpretable rows with coherent timings; old selectors still produce matching predictions under the old protocol.

---

## P8 — Experiment execution matrix and reuse plan

**No GPU training automatically.** Once P0–P7 acceptance gates pass, prepare configs/commands, get authorization and then execute.

### P8.1 Main E02 table (2024kaiyuan Subset 2 + BSTLD)

| ID | Detector | Selection | Run / reuse logic |
|---|---|---|---|
| A1 | YOLO26m, no attention, *untiled* | none | Single pass; clean-split checkpoint and matching evaluator. |
| A2 | YOLO26m, no attention, *tiled* | all | Fixed tiling control; also reused for threshold/all-selected control. |
| A3 | Same tiled YOLO26m as A2 | TileCNN | Trained independently on train split; threshold from validation. |
| A4 | Same tiled YOLO26m as A2 | Eq. (8) param-free | No relevance-head training, threshold from validation. |
| A5 | Same architectural YOLO26m as A2 | Eq. (7) LTS | Train relevance head; if joint training updates detector weights, disclose and add matched detector-only control to isolate this difference. |
| A6 | YOLO26m+CBAM, tiled | all | Matched fixed-tiling attention baseline. |
| A7 | YOLO26m+SE, tiled | all | Matched fixed-tiling attention baseline. |
| A8 | YOLO26m+CBAM, tiled | Eq. (7) LTS | Use matched CBAM detector/features; if jointly updated, compare to matched fine-only control. |
| A9 | YOLO26m+SE, tiled | Eq. (7) LTS | Paper's complete variant; if jointly updated, compare to matched fine-only control. |
| B1 (extra) | Same tiled YOLO26m as A2 | Independent GridNet | Existing baseline; cannot stand in for A5. |

**Important experimental design subtlety:** Truly joint optimization changes the detector weights. Hence an A2 vs A5 comparison may mix **selector benefit** and **backbone fine-tuning benefit** if A5 uses a newly updated YOLO detector. Add either (i) a frozen-backbone LTS diagnostic where the detector is held fixed, and/or (ii) a matched “jointly trained YOLO, all tiles selected” control. Keep manuscript primary rows but disclose this control in supplementary ablation. This is necessary to make claims about LTS causal rather than simply descriptive.

For each dataset collect `mAP50`, `mAP50-95`, recall, FPS (Table II), plus extra diagnostic metrics and seed variability if feasible. **Never force a positive result**.

### P8.2 Reuse completed models for other manuscript tables

| ID | Dependency | Procedure |
|---|---|---|
| E01/Table I | A9 | Run final evaluated LAT-YOLO with identical published SOTA protocol where possible. Avoid mixing differing evaluation procedures. |
| E03/Table III | Trained A9 | Sweep `δ = 0.3, 0.5, 0.7, 0.9` on validation for trade-off, then evaluate the predeclared sweep on test if paper reports it; no retraining for thresholds. Compare all-tile control. |
| E04/Table IV | A6–A9 | Report CBAM/SE fixed vs LTS, parameter delta and FPS; no duplicate training if weights/protocol match. |
| E05/Table V | Final method | Add YOLOv8m and YOLO11m feature providers + trained LTS with backbone-specific layer selection. |
| E06/Table VI | Valid UAV splits | On VisDrone/SeaDronesSee run baseline, fixed, A9; evaluate `AP_S`. |
| E07/Figure 2 | Predictions+relevance maps | Ground truth vs untiled vs fixed vs LAT+heatmap + selected tile overlay on traffic and aerial images. |
| E08/Section V-F | Profiles from P7 | Coarse, relevance, fine, NMS, total; matched fixed. Include average selected tiles and break-even analysis. |

### P8.3 Runtime/accuracy sweeps: recommendations beyond manuscript

These are **optional proposals**, not presently required measured results: tile size/overlap sensitivity, feature-stage P2/P3/P4 selection, matched keep-rate comparisons against GridNet, λ sensitivity, min_keep=0 vs safety floor, max vs mean spatial aggregation (new variant; not paper default), and performance vs object density.

### P8.4 Output contract

Store runs in unique paths, e.g.:

```text
outputs/lat_yolo/<dataset>/<exp_id>/<architecture>/<run_id>/
  resolved_config.yaml
  dataset_split_manifest.json
  environment.json
  checkpoint_fingerprints.json
  per_image_predictions.json
  per_image_metrics_and_timing.jsonl
  summary.json
  figures/
```

All generated file paths and metadata must reflect actual paths; this is a recommended schema, not a preexisting structure.

---

## P9 — Final documentation and reporting audit

- [ ] Update `README.md`, `RUNBOOK.md`, `HANDOFF.md` to distinguish previous GridNet research from new LAT-YOLO feature-guided method. Do not erase the historical GridNet story.
- [ ] Document exact difference: GridNet uses 256×256 independent CNN + center-cell labels; LTS uses YOLO shared feature map + one 1×1 projection + GT-overlap targets.
- [ ] Record all code/model deviations from Section III of paper; ensure Methods describes **what was actually implemented and trained**.
- [ ] Replace every `[value]`, `[]` and `[0.97]` in the manuscript **only with verified results**, or leave honestly marked pending.
- [ ] Ensure Table II A3/A4/A5/B1 comparisons are matched on detector and tiling; Figure 2 is based on actual detections.
- [ ] Confirm that reported “C+1 added parameters” refers only to **LTS relevance head**, not total model/method FLOPs or coarse-pass cost.
- [ ] Confirm that speed statements mean improvement over **fixed tiling** unless there is evidence of improvement over untiled YOLO.
- [ ] Clearly distinguish frozen pilot from joint-method final numbers. If joint training is not implemented, revise the paper rather than misrepresenting the experiment.
- [ ] Check complete provenance and cross-reference generated metrics to source commit, checkpoint SHA, dataset split, and config.

## 3. Proposed file change checklist (agent handoff)

This table is a **to-do roadmap**, not evidence that these paths already exist. Keep additions minimal; reorganize only when required.

| Path in `pipeline/` | Status | Work |
|---|---|---|
| `src/adaptile/models/coarse_features.py` | NEW | Shared-backbone feature provider and geometry metadata. |
| `src/adaptile/models/lts.py` | NEW | Single learnable 1×1 projection (`C+1` params). |
| `src/adaptile/relevance_targets.py` | NEW | GT-overlap feature-map supervision; do not create `adaptile/data/` because `adaptile/data.py` already exists. |
| `src/adaptile/selectors/relevance_utils.py` | NEW | Original->feature coordinates and max tile scoring. |
| `src/adaptile/selectors/lts.py` | NEW | `lts` selector, loads matched head checkpoint. |
| `src/adaptile/selectors/param_free_avg.py` | NEW | Eq. (8) selector. |
| `src/adaptile/training_lts.py` | NEW | Frozen pilot + real joint loss training. |
| `scripts/train_lts.py` | NEW | Config-driven training entry point. |
| `configs/train/lts_kayuan.yaml` | NEW | Versioned train configuration (no fake values/paths). |
| `configs/eval/lts_*.yaml` | NEW | Matched ablation/threshold configs. |
| `src/adaptile/detectors.py` | EDIT | Expose shared model and safe coarse feature extraction binding. |
| `src/adaptile/pipeline.py` | EDIT | Optional detailed selection output, safe empty tiles, full latency. |
| `src/adaptile/selectors/__init__.py` | EDIT | Register new types. |
| `src/adaptile/selectors/base.py` | EDIT IF NEEDED | Backward-compatible richer selection API. |
| `src/adaptile/config.py` | EDIT | Validate new selector/model arguments and provenance fields. |
| `scripts/evaluate.py` | EDIT | Construct selectors with shared detector and export E02/E08 measurements. |
| `src/adaptile/metrics.py` | EDIT / EXTEND | Validate metrics; provide AP_S if requested by dataset. |
| `src/adaptile/models/attention.py` | AUDIT/EDIT | Register all attention parameters before optimizer starts. |
| `configs/models/yolo26m-modified.yaml` | AUDIT/EDIT WITH VERSIONING | Resolve P5 inconsistency; do not overwrite historic architecture identity. |
| `src/adaptile/training.py` | EXTEND MINIMALLY | YOLO architecture variants/training support. |
| `test/test_lts_*.py` | NEW | Math, geometry, shared features, integration, gradients, checkpoint, timings. |
| `scripts/data/*` | NEW AS NEEDED | Group-disjoint splitting and leakage audit. |
| `README.md`, `RUNBOOK.md`, `HANDOFF.md` | EDIT LAST | Accurate new method/handoff instructions. |

## 4. Required minimal tests before authorizing full experiments

Agent must report actual outcomes, not check items optimistically:

```bash
# From the 'pipeline/' working directory (illustrative; only run after files exist)
python -m pytest test/test_lts_math.py test/test_relevance_geometry.py -q
python -m pytest test/test_coarse_features.py test/test_lts_pipeline.py -q
python -m pytest test/test_lts_training.py test/test_attention_registration.py -q
python -m pytest test/ -q
```

**Smoke test goals:**

1. Eq. (7) exact parameter count; Eq. (8) exact order of operations.
2. `image -> backbone -> relevance map -> tiles` works end-to-end on synthetic or available data.
3. Tile score and GT map align with resize/letterbox; no index errors on odd, empty, padded or non-square inputs.
4. Full-selection parity vs fixed tiling (same detector) within numerical tolerance.
5. Zero-selection path is safe and logs zero selected tiles.
6. LTS pilot optimizer updates 1×1 head; joint optimizer updates intended shared detector/backbone weights with proper loss components.
7. GridNet/TileCNN/none backward compatibility.
8. Repeated runs produce matching metrics and stable timing semantics.
9. Experiment manifests explicitly show split integrity, model SHA, config/feature stage and training mode.

## 5. Pause points: ask the user for decisions instead of guessing

The agent should explicitly request confirmation **before long training** on:

- Which **clean Kayuan group split** and exact Subset 2 version to use; whether user has video sequence IDs and BSTLD/VisDrone data.
- Whether to **remove P5** in the final YOLO26m architecture or correct the manuscript description to match current checkpoint architecture. This may require retraining every affected model.
- Which single backbone feature stage and coarse preprocessing policy the pilot will test (recommend a short validation-only stage sensitivity test; do not infer from paper).
- Whether final study requires fully **joint optimization** as Section III says (default yes), or whether author plans a clearly labeled two-stage/frozen-head variant.
- Primary tile size/overlap, λ and threshold selection protocol, train seeds, available GPU/time budget.
- Whether optional `min_keep` failsafe is allowed (default paper-faithful 0).

The agent may implement configurable infrastructure without waiting for all choices; it must not finalize checkpoints, manuscript claims or final tables without them.

## 6. Deliverables from the coding agent

For each completed phase, return the following concise handoff format:

```markdown
### Phase <ID> — <title>
**Status:** implemented / tested / blocked
**Files changed:** exact paths and short purpose
**Paper alignment:** equations/sections implemented; known departures
**Commands run:** exact commands, environment, actual results
**Artifacts:** outputs/configs/checkpoints with paths and hashes
**Outstanding decisions:** user choices, risk, next-phase gate
**Impact on experiments:** E02 rows now runnable; which results still invalid
```

**Final definition of done:** The repository can evaluate `none`, `tile_cnn`, `gridnet`, `param_free_avg`, and genuinely shared-backbone `lts` on a matched clean dataset; the joint training path is real and test-proven; timing includes coarse overhead; all E02 rows can be produced without fabricated results; prior work remains preserved; and the manuscript's implementation description matches actual code.
