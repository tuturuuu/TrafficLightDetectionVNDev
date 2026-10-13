# LAT-YOLO Phase P0 audit and proposed protocol

Audit date: 2026-10-12. Baseline commit: `f7c1db150c23ceea1828d1abd0d95e7fc96a27ec`.

**Status: audit verified with blockers; P0 approval gate open. LTS is proposed, not implemented.**
Only new audit artifacts were added. No existing source, configs, datasets, weights,
historical results, branch, or Git index were changed. No training or GPU inference ran.

## Evidence and checkout

- Working root is `/home/anhld/AdaptiveTiling/TrafficLightDetectionVNDev`, not a nested `pipeline/` directory.
- Branch is `lts-applied`, tracking `origin/lts-applied`; local `refactor/pipeline` is at `6dcbd5c1`. There is no local branch named `refactor`. Remote refs were inspected locally, not fetched.
- Pre-existing tracked change: `configs/train/detector_kayuan.yaml:35`, `train.device: 0 -> 1`. `spec.md` and `implementation_plan.md` are pre-existing untracked files. Preserve all three.
- The specification describes an older revision. Current HEAD already includes sequence-split configs, YOLO26m P2/SE/CBAM architectures, explicit GridNet validation paths, and latency/memory reporting. These should be audited and extended, not reimplemented from the older baseline.
- `baseline_inventory.json` records source/config hashes, 47 checkpoint hashes and loaded metadata, 19 run configs/log summaries, 15 historical output/document artifacts, dataset manifests, versions, and hardware. Checkpoint timestamps and filesystem mtimes are separate fields; neither substitutes for a split manifest.
- `LAT_YOLO.pdf` was not found in the repository or the searched attachment paths. Manuscript equations below come from the supplied specification, not direct PDF verification. The exact Subset 2 source/version and dataset licenses are still unverified.

## Findings, ordered by impact

### F01: SeaDrone sequence dataset cannot currently run

All **10,477 image symlinks and 10,477 label symlinks** under `data/seadrone_sequence`
are broken: train 7,312; val 1,579; test 1,586. Targets refer to the absent
`data/SeaDroneSees_70_15_15` directory. The manifest's group intersections are empty,
but that does not make the on-disk dataset usable. `configs/train/selector_seadronesee.yaml`
and the detector/eval configs use this root.

Proposed P1 action: verify filename, annotation and content identity against the
available original `data/SeaDroneSees`, then create a new versioned dataset or
restore the exact referenced source. Do not silently repoint symlinks based on
basenames alone or resplit an official unlabeled test set.

### F02: Old weights invalidate clean-split accuracy claims

`configs/eval/kayuan.yaml:13` selects sequence-test images but `:35` still loads
the old YOLOv8-CBAM detector, and `:25` loads the old GridNet. The SeaDrone config
has the same clean-data/old-weight provenance issue.

The Kayuan manifest contains **2,409 / 3,035 test images** and **2,428 / 3,036 val
images** whose source is the old train directory. For SeaDrone's saved manifest,
the corresponding counts are 1,123 / 1,586 and 1,095 / 1,579. This is direct frame
overlap, not just similar video content. Existing detector metadata references
old datasets (`label2_unzipped`, `data/kayuan`, or the old SeaDrone splits).
Four YOLO26m-P2 timing-run checkpoints exist under `runs/eval2`, but they also
used the old Kayuan split; they are not clean final baselines.

Preserve these checkpoints for regression, historical reproduction, or explicitly
contaminated engineering pilots. No clean-split task-trained detector or selector
was established by this audit. Final models must be trained under an approved
clean protocol. COCO weights are possible initialization assets, not final results.

### F03: Required data tools are missing from version control

`.gitignore:32` uses `data/`, which also ignores `scripts/data/`. Locally present
`check_split_leakage.py`, `convert_visdrone_to_yolo.py`, and `prepare_bstld.py` are
untracked and ignored. The first two are imported at collection time by
`test/test_handoff_fixes.py:13` and `test/test_visdrone_conversion.py:10`.
Therefore the local passing suite does not establish that a fresh clone can run it.

Proposed P1 action: scope the dataset ignore rule to `/data/`, review the existing
local tools, retain their contents, and include the required source in the later
reviewed change. No files were staged in P0.

### F04: LTS and its parameter-free control remain absent

The registry contains `none`, `gridnet`, and `tile_cnn`. GridNet is an independent
256-square CNN with center-cell supervision and two head convolutions. TileCNN
scores individual resized tiles. Neither reads YOLO backbone features.
There is no shared coarse provider, one-convolution LTS head, overlap-target
generator, matched LTS checkpoint loader, or joint trainer.

`scripts/evaluate.py:121` constructs the detector before selectors, which supports
explicit dependency injection. `Pipeline.from_config` currently constructs the
selector before the detector (`src/adaptile/pipeline.py:52`), so both entry paths
need compatible binding. Empty crops already return no detections in
`YOLODetector.__call__`; preserve that behavior. GridNet's fallback must remain
unchanged while new LTS defaults to `min_keep=0`.

### F05: Architecture availability improved, but manuscript conflict remains

All five `ARCHS` entries constructed and forwarded on CPU under Ultralytics 8.4.32.
YOLO26m-P2, SE and CBAM all detect at strides **4/8/16/32**. SE and CBAM have
matched attention positions 5/8/11, with P3/P4 skip connections taken before the
attention modules. P5 still exists. The supplied specification's claim that
YOLO26m+CBAM is missing is stale; its concern about removing P5 is still relevant.

`HANDOFF.md:134` records an earlier user choice of YOLO26m-P2 for all E02 rows.
Carry this forward as the intended reference architecture. Removing P5 would be
a separately versioned architectural change requiring matched retraining; it
must not overwrite existing YAML identities. Resolve the paper wording before
claiming the deepest stage was removed.

The historic YOLOv8-CBAM model is not stock YOLOv8m plus attention alone: its
strides are **4/8/16**, versus stock YOLOv8m's **8/16/32**. Existing cross-model
results cannot isolate only an attention contribution.

### F06: Lazy attention is a conditional risk, not a proven optimizer omission

`src/adaptile/models/attention.py:39` and `:89` create parameters during forward.
However, installed `DetectionModel.__init__` performs a 256-square stride forward
before returning. All attention parameters were present in an optimizer created
after YOLO construction, and the explicit CPU forward added zero parameters.
Do not assert that previous models necessarily failed to train their attention.

Standalone first-forward/device/dtype behavior remains unsafe without explicit
initialization: CBAM creates CPU/default-dtype layers; SE matches device but not
input dtype, and small channel counts can yield zero hidden channels. P6 should
add focused device, dtype, optimizer, gradient and serialization tests and a
versioned initialization path, preserving legacy pickle aliases and parameters.

### F07: Evaluation improvements are real but incomplete for the new protocol

Already implemented: class-aware mAP50/mAP50-95, micro precision/recall, original-image
wall FPS, select/detect/merge/total per-image latency, p50/p95, THOP complexity,
and peak allocated GPU memory (`scripts/evaluate.py:54`).

Still unresolved:

- Warmup runs only the first two tiles, not the representative complete path (`src/adaptile/pipeline.py:89`).
- `detect_s` combines cropping, detector preprocessing/forward/postprocessing and remapping. There is no separate coarse/head/scoring decomposition yet.
- All requested selectors are resident together, so peak allocated memory is process-level, not necessarily an isolated method footprint. `_sync()` synchronizes all visible CUDA devices; detector and selector devices can differ.
- AP uses the project's 101-point evaluator, not a validated official COCO/VisDrone protocol. Tests compare to legacy/local evaluators, not identical exported predictions evaluated by a trusted external implementation. AP_S and ignore-region handling are absent.
- `conf=0.25` limits the predictions used for AP. Per-tile YOLO NMS/end-to-end behavior and max detections are not explicit in the detector config, while cross-tile NMS is separate.
- Output lacks checkpoint/split/code/environment fingerprints and serialized per-image predictions. P50/p95 exist, but timestamps, sample identities and unrounded summary values need an expanded versioned output contract.
- `data.py:26` treats missing labels as empty; tiling also accepts missing labels. Fail-fast data preflight is needed before scientific runs. Official SeaDrone test has 3,750 images without local labels; it cannot yield local ground-truth metrics as-is.

Keep the historical evaluator/protocol identifiable while adding a validated
research protocol. Do not silently combine outputs from the two protocols.

### F08: Config and documentation do not consistently identify experiments

`configs/train/detector_kayuan.yaml:24` selects **YOLO26m+CBAM**, but its header
and output name (`:41`) still say YOLOv8-CBAM. Use new versioned experiment configs
with truthful architecture and split IDs; leave the user's GPU-device edit intact.

`HANDOFF.md:135` references `doc/kayuan_ablation_A1_A9.md` and
`configs/experiments/kayuan_p2/`, neither of which exists in this checkout.
The root README still says VisDrone and YOLO26m attention variants are unavailable.
`RUNBOOK.md:15` incorrectly equates kept grid cells with kept tiles; overlapping
tile footprints invalidate that equivalence. Missing a cell does not guarantee
the object is outside every selected overlapping tile. Its 1920x1080, tile-640,
overlap-0.2 example is 8 candidate tiles in current code, not approximately 6.

`prepare_detector_data` uses the YAML's parent rather than its `path:` value.
Current sequence YAML roots happen to agree, but general dataset support needs
proper path resolution. Tiling cache validation checks size/overlap/visibility,
not source-manifest identity or complete output contents. The manual tiling CLI
does not produce the same cache marker. Preserve old behavior for reproduction;
introduce validated versioned data preparation for new runs.

### F09: Remaining data and environment provenance

| Dataset | Original images train / val / test | Current status |
| --- | --- | --- |
| Kayuan sequence | 14,127 / 3,036 / 3,035 | Files accessible; manifest groups 53/10/10, intersections empty |
| SeaDrone sequence | 7,312 / 1,579 / 1,586 entries | All image and label targets broken |
| VisDrone | 6,471 / 548 / 1,610 | Image and YOLO-label files accessible; report identifies test-dev |
| BSTLD | Not established | A 4,786,803,364-byte POSIX TAR exists; no usable dataset YAML; contents not audited/extracted |

Kayuan group keys use filename timestamps; this establishes disjoint recorded
sequence IDs, not disjoint locations or absence of near-duplicate content.
Full annotation checks, content duplicates, split class distributions, source
variant confirmation and license verification remain P1 work.

Environment: Python 3.11.15, Torch 2.9.1, Torchvision 0.24.1, Ultralytics 8.4.32,
CUDA build 12.8, two RTX 4090 GPUs. Both GPUs were busy at the audit snapshot.
Installed NumPy 2.3.3 and OpenCV 4.14.0.94 differ from requirements pins 2.2.6 and
4.12.0.88. Record the actual environment before freezing a reproducible one;
do not upgrade packages during a baseline audit.

## Decisions carried forward and decisions still open

**Required by the supplied specification:** preserve all old selector modes;
use the same YOLO weights/model for coarse and fine paths; exact one-channel
1x1 projection with C+1 parameters; sigmoid before channel mean for the control;
max over overlapping feature cells; strict score > threshold; min_keep=0;
GT-overlap targets; separately labeled frozen pilot followed by real joint
detector-plus-relevance optimization. Joint optimization is required, not an
optional substitution. Include an all-tiles control using the jointly updated
detector to separate selection gains from detector fine-tuning gains.

**Recorded earlier, carried forward:** YOLO26m-P2 architecture family across
the primary table; Kayuan label2/16-class data as the available candidate;
validation-only selection of hyperparameters. Exact official Subset 2 identity
still needs source confirmation.

| Decision | P0 disposition |
| --- | --- |
| Working branch | Propose `feature/lat-yolo-lts` from current `f7c1db15`, retaining the newer fixes and local device edit. Await approval because the supplied P0 instructions explicitly pause on a dirty/different branch. |
| Actual manuscript | Need the path or document for `LAT_YOLO.pdf`; do not certify paper alignment from the spec alone. |
| P5 stage | Preserve existing P2-P5 architecture for the pilot per recorded user choice; confirm whether manuscript will be corrected or a distinct no-P5 architecture is intended. |
| Coarse feature | Experimental pilot proposal: one post-stage P3 tensor (stride 8), after attention when present. Exact layer/channel metadata must be verified by P2 tests; never assume indices transfer between architectures. P2/P4 remain optional validation ablations. |
| Coarse preprocessing | Proposed pilot: aspect-preserving YOLO-compatible 640-square letterbox, RGB, /255; store exact scale, integer padding and cell footprint mapping. Not a manuscript-established value. |
| Tiles and fine input | Proposed matched starting point: tile 640, overlap 0.20, fine imgsz 640, batch 16, existing crop/NMS semantics; freeze final settings after validation. |
| Loss weight and imbalance | No approved lambda or pos_weight. Expose both, log positive-cell prevalence, tune only on val; do not transfer GridNet's tuned pw8/pw10 as an LTS fact. |
| Threshold | Validation-only sweep and predeclared accuracy/recall tolerance; compare param-free and learned selectors at matched keep fraction as well as threshold curves. The final tolerance and threshold are not selected in P0. |
| Sampling/joint training | Pair original-image coarse labels with corresponding detector training crops; implement shared gradients, optimizer membership and matched checkpoint restoration in P5. No inference dependence on GT. |
| Seed/budget | Seed 0 engineering pilot; final seed matrix and GPU time authorization remain open. Earlier main-only three-seed plan and current three-seed configs disagree. No long training is authorized by this audit. |
| Initialization | Current Kayuan config is scratch. Scratch versus COCO for final matched baselines remains to be fixed with architecture and budget; old contaminated task weights are not clean final initialization. |
| Visibility/padding | Proposed: any valid target-class box overlapping a nominal feature cell is positive; mask padding-only cells from relevance loss. Define boundary clipping and partially padded cells in geometry tests. |

## Proposed incremental modifications

| Phase | Concrete work after the relevant gate |
| --- | --- |
| P1 | Fix source-tool tracking; audit/restore versioned data; add file/label/duplicate and manifest preflight; record checkpoint eligibility; validate metrics/AP_S and YOLO26 postprocessing; add accurately named configs without overwriting historical ones. |
| P2 | Add `models/coarse_features.py` with graph-aware early termination and exact preprocessing metadata; expose the existing detector instance. Account for Ultralytics predictor initialization/fusion before binding; verify stage identity, gradients and non-square geometry. |
| P3 | Add `models/lts.py`, `relevance_targets.py`, `selectors/relevance_utils.py`; test C+1, mean(sigmoid), GT overlap, max scoring, strict thresholds and empty regions. |
| P4 | Add `selectors/lts.py` and `selectors/param_free_avg.py`; extend registry/config/pipeline/evaluator via backward-compatible detector binding; preserve existing selector defaults and assert keep-all parity/zero-selection safety. |
| P5 | Add `training_lts.py`, `scripts/train_lts.py` and versioned configs; frozen pilot then actual YOLO detection loss plus relevance loss, gradient isolation, resume/EMA and matched detector/head checkpoint tests. |
| P6 | Extend attention registration tests and versioned model definitions only where needed. Three matched P2-P5 variants already exist; do not recreate them or alter old checkpoints implicitly. |
| P7 | Extend existing latency fields with coarse/preprocess/head/scoring/fine/remap stages, full warmup, isolated memory accounting, per-image predictions, hashes and validated AP_S; matched E02 collector. |
| P8 | Prepare matched A1-A9 plus GridNet B1 and joint-detector all-tiles controls; request approval before long training/evaluation. |
| P9 | Update README/RUNBOOK/HANDOFF and paper-to-code mapping after verified milestones; preserve historical results and never fill result placeholders from assumptions. |

Installed API review confirms graph edges in `BaseModel._predict_once` and a
YOLO26 `E2ELoss` criterion. It does not prove that the proposed joint trainer
works. No LTS loss API, gradient path, feature identity or speed claim was
verified in P0; those are explicit later acceptance gates.

## Verification and reproduction

Baseline test command, from repository root:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/traffic_light/bin/python -m pytest test/ -q -p no:cacheprovider \
  --junitxml=/tmp/lat_yolo_p0_tests.xml
```

Result: **174 passed, 13 skipped, 7.30 seconds**. The skips are GPU/data-dependent
parity and opt-in full/training checks. They are not evidence of GPU prediction
parity or training correctness. `baseline_tests.xml` preserves the result.
`git diff --check` passed on the pre-existing tracked diff.

Inventory reproduction (use a new destination; existing inventory is never overwritten):

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/traffic_light/bin/python \
  outputs/audits/p0_f7c1db15_20261012T141320Z/audit_probe.py \
  --out /tmp/lat_yolo_p0_reaudit --pytest-xml /tmp/lat_yolo_p0_tests.xml
```

The inventory probe loaded all 47 local checkpoints on CPU and verified five
architecture forward/optimizer-membership probes. It did not execute an optimizer
step, a training loss, dataset inference, or a performance benchmark.

**Next-phase gate:** confirm the branch departure from the old plan, supply the
manuscript location, and settle/record the architecture discrepancy. Data failures
and protocol provenance must be resolved before final research experiments.
No new E02 LTS row became runnable from this documentation-only P0 milestone.
