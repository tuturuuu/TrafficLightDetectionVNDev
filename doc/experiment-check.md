# Table II — Kaiyuan CD

Seeds: **0, 42, 100**. Scope: **training only**; test evaluation is performed
manually by the user. Approved recipe: detector 100 epochs, joint LTS 50 epochs,
with existing early stopping. Runtime: conda `LAT-YOLO`, tmux `kaiyuan`, physical
GPU **1** (second GPU).

Training completed and independently verified on **2026-10-23 10:18 UTC**, with the user's updated **Baseline + SE**:

| Configuration | Tiling | Selection | Attention | Seed 0 | Seed 42 | Seed 100 |
|---|---|---|---|---|---|---|
| A1 Baseline + SE | – | – | SE | Verified | Verified | Verified |
| A2 Fixed tiling | Fixed | – | – | Verified | Verified | Verified |
| A3 Naive filter | Fixed | CNN | – | Verified | Verified | Verified |
| A4 Param-free avg | Grid | Avg. | – | Ready via A2 | Ready via A2 | Ready via A2 |
| A5 LTS learned | Grid | LTS | – | Verified (50 epochs) | Verified (50 epochs) | Verified (50 epochs) |
| A6 Fixed + CBAM | Fixed | – | CBAM | Verified | Verified | Verified |
| A7 Fixed + SE | Fixed | – | SE | Verified | Verified | Verified |
| A8 LTS + CBAM | Grid | LTS | CBAM | Verified (50 epochs) | Verified (49 epochs, early stop) | Verified (50 epochs) |
| A9 LTS + SE | Grid | LTS | SE | Verified (50 epochs) | Verified (49 epochs, early stop) | Verified (50 epochs) |

At the user's request, A1 is **Baseline + SE**, using the existing no-tiling SE
detectors; no new baseline is trained. This replaces the PDF's no-attention
baseline and must retain its explicit SE label when reported.
The old `frozen_head_pilot` run is not the joint learned method. Param-free averaging requires
no additional training. No test metrics have been entered in this checklist.

All **27/27** training cells are ready, backed by **24 verified training assets**:
12 existing detectors (including three reused Baseline + SE models), three
existing CNN selectors and nine completed joint LTS runs. No jobs remain queued.
The final trainer exited with code 0, the queue completed, and physical GPU 1
was released. The `kaiyuan` tmux session remains available.

CBAM seed 42 and SE seed 42 each stopped after 49 epochs with `bad_epochs=10`
(configured patience 10); both selected their respective epoch 39 paired
checkpoints. The other seven joint LTS runs completed 50 epochs.

Artifact hashes and exact best checkpoint paths are recorded in
[`audit.json`](../outputs/lat_yolo/kaiyuan_cd/table2_train/audit.json).
The independent [completion audit](../outputs/lat_yolo/kaiyuan_cd/table2_train/completion_audit.json)
verified all seeds, terminal states, training/source provenance, dataset hashes,
checkpoint pairing and 27 resolved manual-test configs. No test evaluation was
run; no Table II test output JSON existed at completion.

See [training protocol, artifact mapping, monitoring and manual test commands](kaiyuan_table2_train.md).
