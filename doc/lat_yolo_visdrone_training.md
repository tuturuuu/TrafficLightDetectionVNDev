# LAT-YOLO on VisDrone

The pipeline trains YOLO26m P2–P5 + SE on uniform tiles, then jointly trains
the detector and LTS using original frames and sampled fine tiles.

| Setting | Detector | Joint LTS |
|---|---|---|
| Maximum epochs / patience | 100 / 30 | 100 / 20 |
| Batch size | 8 | 8 |
| Workers | 8 | 8 |
| Image size | 640 | Fine 640, coarse 640 |
| Seeds | 0, 42, 100 | 0, 42, 100 |
| Initialization | Scratch | Best stage-1 detector |
| Physical GPU | 1 | 1 |

Tiles are 640 pixels with 20% overlap and minimum GT visibility 0.3. LTS
uses layer 5 (P3 after SE), with lambda=1 and pos_weight=1. These are starting
settings from the repository, not tuned VisDrone optima. Both stages use AMP.
Eight workers are used on the 80-thread host; Ultralytics uses 16 for validation.
The launcher's snapshots preserve the configured early-stopping patience.

The existing dataset has 6,471 train, 548 validation and 1,610 test images,
with 10 classes. Training uses train/validation only. The content inventory
is `data/visdrone/split_manifest_lat_v1.json`; it preserves existing splits.

Run from the repository root with the installed `traffic_light` environment:

```bash
bash scripts/train_lat_visdrone.sh runs/lat_yolo/visdrone/my_new_run
```

The output directory must not exist. The launcher snapshots both configs,
records `pipeline.pid`, `plan.json` and `status.txt`. Each seed runs sequentially:
detector, joint LTS with a 100-epoch schedule, then validation at thresholds
0.3 and 0.5. A stage can stop early when its configured patience is reached.
Logs are `detector_seedN.log`, `lts_seedN.log`, and `val_seedN_thr03/05.log`.
The launcher sets `CUDA_VISIBLE_DEVICES=1`; local device 0 in the snapshots
therefore means physical GPU 1. The source YAMLs also support direct runs
with device 1 when CUDA devices are not remapped.

Stage-1 weights are `yolo26m_p2_tiled_se_seedN/weights/best.pt`. For each final
LAT-YOLO model, read `lts_joint_se_seedN/best.json` and use both `detector.pt`
and `lts.pt` from the indicated `best_epochNNNN` directory. Joint training
updates the detector, so its LTS head must use that matching detector.
`resume.pt` contains training state, not inference weights.

The sweep evaluates all 548 original validation images using both uniform
tiling and LTS on the same joint detector. Outputs in `validation/` contain
per-seed JSON, predictions, resolved YAML and mean/sample-SD summaries for
each threshold (`summary_thr03.json/.csv`, `summary_thr05.json/.csv`).
The LTS training threshold remains a diagnostic value of 0.5; sweep thresholds
are applied during validation, without retraining for each threshold.

No test-set metrics or official VisDrone benchmark results are produced by this
launcher. Validation uses the existing project COCO adapter on converted labels.

## Existing run split across GPUs

The run `se_3seeds_20261024_161143` was rescheduled on 2026-10-26:
physical GPU 1 runs seeds 0 and 42; physical GPU 0 runs seed 100. Each seed
still runs detector, joint LTS, and both validation thresholds. The original
detector process for seed 0 continues without restarting its current epoch.

`queue.json` records the current allocation and worker PIDs. `queue_gpu0.json`
and `queue_gpu1.json` record each queue's current stage and child PID;
`status.txt` summarizes both. Queue logs are `queue_gpu0.log` and
`queue_gpu1.log`; training/evaluation logs and checkpoints retain their
original per-seed paths. `completed_seeds.txt` and `seedN.completed.json`
are written after that seed's full sweep. The last queue to finish aggregates
all three seeds into the existing validation summary paths.

The controller is `scripts/runtime/lat_visdrone_queue.py`. It sets
`CUDA_VISIBLE_DEVICES` independently for each queue; the snapshots keep
local device 0. Per-seed locks prevent duplicate execution. To hand off the
live seed-0 detector, only the original shell controller is stopped; its
detector and data-loader processes continue. The replacement controller
reads the detector's actual exit status before advancing to joint LTS,
then retires the original shell so it cannot schedule seed 100 again.

`monitor_pipeline.py` in the run directory supports both queues. This split
preserves the existing P2–P5 + SE architecture and training recipe; it does
not change the experiment into an exact reproduction of the PDF.
