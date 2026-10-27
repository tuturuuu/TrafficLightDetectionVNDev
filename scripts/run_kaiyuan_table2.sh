#!/usr/bin/env bash
set -euo pipefail
cd /workspace/LAT-YOLO
source /opt/conda/etc/profile.d/conda.sh
conda activate LAT-YOLO
# UUID prevents device-order ambiguity; cuda:0 below is PHYSICAL GPU INDEX 1.
expected_gpu=GPU-7459ff17-edbb-d846-2f31-c5af0ae49739
actual_gpu=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader)
[[ "$actual_gpu" == "$expected_gpu" ]] || { echo 'Physical GPU 1 UUID changed'; exit 1; }
export CUDA_VISIBLE_DEVICES="$expected_gpu"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p outputs/lat_yolo/kaiyuan_cd/table2_train
python -u scripts/kaiyuan_table2.py run 2>&1 | tee -a outputs/lat_yolo/kaiyuan_cd/table2_train/queue.log
