#!/usr/bin/env bash
# Eval 2 (plan Notion, bước 2): đo thời gian/epoch và mAP sau 5 epoch của YOLO26m-P2
# trên Kayuan, có/không pretrained COCO, có/không tiling. Dùng để ước lượng lại bước 3.
#
# Mỗi GPU chạy một hàng đợi 2 run, nối tiếp nhau. Trước mỗi run, chờ tới khi GPU đó
# TRỐNG (memory < MAX_MEM_MIB và util < MAX_UTIL% liên tục CHECKS lần, cách nhau
# INTERVAL giây) để thời gian đo không bị job khác làm sai.
#
#   bash scripts/eval2_timing.sh            # chạy (nên chạy trong tmux/nohup)
#   bash scripts/eval2_timing.sh --dry-run  # chỉ in lệnh
#
# Kết quả: runs/eval2/<name>_seed0/results.csv ; log: runs/eval2/logs/*.log
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/anaconda3/envs/traffic_light/bin/python}
GPUS=(${GPUS:-0 1})
MAX_MEM_MIB=${MAX_MEM_MIB:-2000}
MAX_UTIL=${MAX_UTIL:-10}
CHECKS=${CHECKS:-3}
INTERVAL=${INTERVAL:-60}
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
LOGDIR=runs/eval2/logs; mkdir -p "$LOGDIR"

COMMON=(configs/train/detector_kayuan.yaml
        model.arch=configs/models/yolo26m-p2.yaml model.attention=none
        train.seeds=[0] train.epochs=5 output.project=runs/eval2)

# Hàng đợi theo GPU: tiled trước (quan trọng nhất cho ước lượng), notile sau.
QUEUE_A=("tiled pt" "notile pt")
QUEUE_B=("tiled scratch" "notile scratch")

gpu_free() {  # $1 = gpu id
  local ok=0 mem util
  for ((i = 0; i < CHECKS; i++)); do
    read -r mem util < <(nvidia-smi -i "$1" --query-gpu=memory.used,utilization.gpu \
                         --format=csv,noheader,nounits | tr -d ',')
    if (( mem < MAX_MEM_MIB && util < MAX_UTIL )); then ok=$((ok + 1)); else return 1; fi
    (( i < CHECKS - 1 )) && sleep "$INTERVAL"
  done
  (( ok == CHECKS ))
}

run_one() {  # $1 = gpu, $2 = tiled|notile, $3 = pt|scratch
  local gpu=$1 tile=$2 init=$3
  local ttype=$([[ $tile == tiled ]] && echo uniform || echo none)
  local pre=$([[ $init == pt ]] && echo weights/yolo26m.pt || echo null)
  local name="kayuan_yolo26m_p2_${tile}_${init}_e5"
  local cmd=("$PY" scripts/train_detector.py "${COMMON[@]}"
             tiling.type=$ttype model.pretrained=$pre train.device=0 output.name=$name)
  if (( DRY )); then echo "[GPU $gpu] CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"; return; fi
  until gpu_free "$gpu"; do
    echo "$(date '+%F %T') [GPU $gpu] bận, chờ ${INTERVAL}s để chạy $name"; sleep "$INTERVAL"
  done
  echo "$(date '+%F %T') [GPU $gpu] BẮT ĐẦU $name ($(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader))"
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" > "$LOGDIR/$name.log" 2>&1 \
    && echo "$(date '+%F %T') [GPU $gpu] XONG $name" \
    || echo "$(date '+%F %T') [GPU $gpu] LỖI $name, xem $LOGDIR/$name.log"
}

queue() {  # $1 = gpu, rest = runs
  local gpu=$1; shift
  for r in "$@"; do run_one "$gpu" $r; done
}

queue "${GPUS[0]}" "${QUEUE_A[@]}" &
queue "${GPUS[1]:-${GPUS[0]}}" "${QUEUE_B[@]}" &
wait
echo "$(date '+%F %T') Eval 2 xong. Tóm tắt: $PY scripts/eval2_summary.py"
