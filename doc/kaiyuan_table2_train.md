# Kaiyuan Table II training — seeds 0, 42, 100

Scope: the nine configurations in `LAT_YOLO.pdf`, Table II, on
`data/kaiyuan_cd` only. Training is authorized; test evaluation remains manual.
The user subsequently replaced A1 with **Baseline + SE**, reusing the existing
no-tiling SE detectors. A1 is explicitly labeled with attention; it is no longer
the PDF's no-attention baseline. No baseline training is authorized or needed.
The user selected the existing recipe: detector 100 epochs (patience 30),
joint LTS 50 epochs (patience 10), reusing valid trained checkpoints.
These are maximum epochs; configured early stopping is retained.

This recipe differs from the draft PDF's 700-epoch AdamW description. Existing
detectors use Ultralytics `optimizer=auto`, batch 16 and input 640. LTS uses
AdamW, lr 0.001, weight decay 0.0005, lambda 1, positive weight 1, batch 16.
LTS AMP explicitly uses **bfloat16** on the RTX 4090 for all nine joint runs.
The repository backbone is YOLO26m P2–P5; it retains P5, unlike the draft's
description of removing the deepest detection stage. The manuscript needs to
describe the actual protocol before reporting these experiments.

## Completion record

Training finished on **2026-10-23 10:17 UTC**; the independent completion audit
passed at **10:18 UTC**. All **27/27** Table II cells are ready. The controller and
final trainer have exited successfully and physical GPU 1 is free from this
queue. Test evaluation remains manual; no Table II test result JSON was created.

| Joint LTS attention | Seed 0: epochs / best epoch | Seed 42: epochs / best epoch | Seed 100: epochs / best epoch |
|---|---|---|---|
| None | 50 / 45 | 50 / 45 | 50 / 41 |
| CBAM | 50 / 49 | 49 / 39 (early stopping) | 50 / 47 |
| SE | 50 / 44 | 49 / 39 (early stopping) | 50 / 48 |

Both early stops reached the configured patience of 10. Exact paired weight
paths and hashes are in
[`audit.json`](../outputs/lat_yolo/kaiyuan_cd/table2_train/audit.json), with the
full [completion checks](../outputs/lat_yolo/kaiyuan_cd/table2_train/completion_audit.json).
All nine runs used the same training implementation. The first no-attention
seed 0 run recorded the earlier queue script before the Baseline + SE override;
that script is preserved in `table2_train/source_snapshots/`, and the exception
is documented in the completion audit. The training code hashes match.

## Configuration and artifact mapping

Every row below has seeds **0, 42, 100**. Run prefixes are under
`runs/lat_yolo/kaiyuan`; append `_seed<seed>`.

| Row | Configuration | Training artifacts |
|---|---|---|
| A1 | Baseline + SE, no tiling | `yolo26m_p2_notile_se` (reuse only) |
| A2 | Fixed tiling | `yolo26m_p2_tiled_none` |
| A3 | Naive CNN filter | A2 + `weights/lat_yolo/kaiyuan_cd/tilecnn_seed<seed>.pth` |
| A4 | Parameter-free average | A2; no separate selector training |
| A5 | LTS, no attention | `lts_joint_none_pw1_lam1` initialized from A2 |
| A6 | Fixed + CBAM | `yolo26m_p2_tiled_cbam` |
| A7 | Fixed + SE | `yolo26m_p2_tiled_se` |
| A8 | LTS + CBAM | `lts_joint_cbam_pw1_lam1` initialized from A6 |
| A9 | LTS + SE | `lts_joint_se_pw1_lam1` initialized from A7 |

Frozen pilot runs do not satisfy A5/A8/A9. A4 shares the A2 detector and has zero
learned selector parameters. The completed inventory includes nine joint LTS runs. Nine tiled detectors, three no-tiling SE detectors and three CNN
classifiers are reused after audit. Baseline jobs are marked `reuse_only`;
missing or invalid baseline evidence stops the queue instead of training one.

`configs/experiments/kaiyuan_table2/train/` contains 24 explicit recipes, including
the reused model recipes (27 table cells share training dependencies).
`eval_templates/` contains all 27 manual test configurations. The audit
writes ready-to-use versions into `eval/` only once their training dependencies
are verified, resolving each LTS best detector/head pair from `best.json`.

Original split counts: train 800, validation 100, test 100. Train/validation have
6400/800 cached 640-pixel tiles, overlap 0.2, minimum visibility 0.3. Training
reads only train and validation. The original YAML has an old machine path;
`data_table2_local.yaml` supplies the local absolute root for new detector runs
without changing the original YAML or split manifest fingerprints.

## Execution and monitoring

Use conda `LAT-YOLO`, tmux `kaiyuan`, **physical GPU index 1**. The launcher pins
`CUDA_VISIBLE_DEVICES` to that GPU's UUID. Consequently `device=0` / `cuda:0` in
these configs is logical GPU 0, which maps to physical GPU 1. This also pins the
legacy CNN trainer, which otherwise chooses its default GPU. GPU 0 is used by
the separate SeaDronesSee session.

```bash
bash scripts/run_kaiyuan_table2.sh
```

Run this command inside the `kaiyuan` tmux session. It activates the requested
conda environment and runs jobs sequentially, holding a lock against a duplicate
queue. It never overwrites an existing run. Completed artifacts are verified
before skipping. An incomplete LTS run resumes from its own `resume.pt`; an
incomplete detector or CNN run stops for inspection. The queue stops on failure.

```bash
tail -f outputs/lat_yolo/kaiyuan_cd/table2_train/queue.log
cat outputs/lat_yolo/kaiyuan_cd/table2_train/current.json
# Detailed epoch output is in the current job's .log file.
/venv/LAT-YOLO/bin/python scripts/kaiyuan_table2.py audit
```

`audit.json` records per-job checkpoint hashes, completed epochs, exact best
LTS pairs and readiness of all 27 table cells. Completion requires 27/27 ready
and actual terminal training evidence, not just the presence of `best.pt`.
The current job PID in `current.json` is only a pointer: verify it against the
live process before deciding whether the queue is still running.

## Manual test, after training

The queue does **not** run these commands. Choose thresholds on validation before
the final test; template thresholds of 0.5 are defaults, not tuned results.

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate LAT-YOLO
cd /workspace/LAT-YOLO
export CUDA_VISIBLE_DEVICES=GPU-7459ff17-edbb-d846-2f31-c5af0ae49739
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/benchmark_lts.py \
  configs/experiments/kaiyuan_table2/eval/A9_lts_se_seed0.yaml
```

For the Naive filter, Param-free average and LTS learned rows requested earlier,
use these commands after the environment setup above and after the training queue
has released GPU 1. The loop runs test for seeds 0, 42 and 100; it is provided for
manual execution and has not been run by the training queue.

```bash
for seed in 0 42 100; do
  python scripts/benchmark_lts.py "configs/experiments/kaiyuan_table2/eval/A3_naive_seed${seed}.yaml"
  python scripts/benchmark_lts.py "configs/experiments/kaiyuan_table2/eval/A4_avg_seed${seed}.yaml"
  python scripts/benchmark_lts.py "configs/experiments/kaiyuan_table2/eval/A5_lts_seed${seed}.yaml"
done
```

| Method | Test result under `outputs/lat_yolo/kaiyuan_cd/table2_test/` |
|---|---|
| Naive CNN filter | `A3_naive_seed<seed>.json` |
| Param-free average | `A4_avg_seed<seed>.json` |
| LTS learned, no attention | `A5_lts_seed<seed>.json` |

Naive CNN training weights are in
`weights/lat_yolo/kaiyuan_cd/tilecnn_seed<seed>.pth`. Param-free averaging has no
selector checkpoint or selector training output; its detector is the A2 asset
listed above. LTS learned seeds 42 and 100 are already trained; their ready eval
configs select each run's best paired detector and LTS weights automatically.
These JSON test results are produced only when the manual commands run;
`results.csv` and LTS `history.json` contain training/validation records.

Test outputs are `outputs/lat_yolo/kaiyuan_cd/table2_test/<row>_seed<seed>.json`,
with saved YAML and predictions sidecars. They are immutable; a repeat needs a
new `eval.out`. FPS is the full pipeline with profiling disabled. Recall uses
micro IoU 0.5 at detector confidence 0.001; it is not Ultralytics' best-F1 recall
from the training CSV. Use the same evaluator settings for every row.

Joint training changes the detector. The Table II rows evaluate the respective
models; isolating only the inference selection effect additionally requires a
fixed-tiling control on the *same paired joint detector*. Those optional controls
are separate from the nine requested rows.

## Numerical stability investigation

The first FP16 joint run stopped at epoch 2 validation with nonfinite detection
loss. Replaying from epoch 1 reproduced the failure: relevance logits/loss were
finite; detection loss was NaN in FP16 and finite in FP32/BF16 on the same batch.
The failed run is preserved under `runs/lat_yolo/kaiyuan/failed_attempts/` and the
diagnosis is logged in `outputs/lat_yolo/kaiyuan_cd/table2_train/precision_diagnosis.log`.
It is not counted as a completed experiment.

The trainer now supports explicit `train.amp_dtype`, and advances the learning
rate scheduler only when GradScaler actually performs an optimizer update.
BF16 does not require gradient scaling. Nonfinite unscaled BF16/FP32 gradients
fail explicitly. Six CPU tests passed, including a real scaler-overflow
regression test and exact interrupted/uninterrupted training parity for both
frozen and joint modes. Validation stability of the restarted real-data run is
monitored separately; passing synthetic tests is not evidence of model quality.
