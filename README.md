# yolo11m_for_label1

This repository contains experiments for traffic-light detection using YOLO (v8/v11/v26), tiled inference/training, optional CBAM custom modules, and a small CNN tile proposal model.

## What each folder does

- `bosch/`: Training/experiments for the Bosch traffic-light dataset (baseline, tiling, and CBAM variants).
- `kayuan2024/`: Training/experiments for the Dataset2/Kayuan-style dataset (baseline, tiling, and CBAM variants).
- `cnn_classifier/`: Small CNN code and weights used for tile proposal/filtering before YOLO.
- `evaluations/`: Evaluation scripts, metrics comparison, visual debugging, and utility analysis scripts.
- `tiling/`: Core tiling and adaptive tiling logic used by training/evaluation pipelines.
- `utils/`: Shared helpers (for example dataset conversion and custom module support).
- `yolo_config/`: Custom model YAML configs (for example CBAM/modified YOLO architectures).
- `runs/`: Ultralytics training/inference outputs (checkpoints, logs, prediction results).

## Model files in root

- `yolov8n.pt`, `yolov8m.pt`, `yolo11n.pt`, `yolo11m.pt`, `yolo26n.pt`, `yolo26m.pt`: pretrained or starting checkpoint files used by experiments.

## Setup

1. Create and activate a Python environment (Python 3.10+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify PyTorch sees GPU (optional but recommended):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Quick run notes

- Most scripts are standalone experiment scripts (for example inside `bosch/`, `kayuan2024/`, and `evaluations/`).
- Start by editing dataset/model paths in the script you want to run, because many scripts contain local absolute paths.
- Example evaluation commands:

```bash
python evaluations/evaluation_without_cnn.py
python evaluations/evaluation_with_cnn.py
```

- Training scripts are in:
  - `bosch/` (Bosch experiments)
  - `kayuan2024/` (Dataset2/Kayuan experiments)

## Notes

- This is an experiment-heavy workspace, so folder names under `runs/` and `new/` represent many trial variants.
- If you use CBAM/custom YAML models, ensure related custom modules/config files are available in `utils/` and `yolo_config/` before training.
