#!/usr/bin/env bash
# Tải Kayuan label2 (train/val/test, images + labels) từ folder Google Drive public,
# không cần đăng nhập, rồi tạo lại split theo video data/kayuan_sequence.
#
# Ví dụ:
#   scripts/data/fetch_kayuan_gdrive.sh                 # key lấy từ .env
#   scripts/data/fetch_kayuan_gdrive.sh --api-key <KEY> --workers 8
# Cần API key (Drive API bật) vì trang public chỉ liệt kê tối đa 5500 file/folder;
# truyền --api-key, đặt biến GDRIVE_API_KEY, hoặc ghi GDRIVE_API_KEY=... vào .env.
# Chạy lại cùng lệnh để tải tiếp file còn thiếu/lỗi.
set -euo pipefail

# 2024kaiyuan/label2-20250121-trafficlight+2345lukou-20295/label2-...-20235
FOLDER_ID="1-poYCQmVfdkW8IA6ZUNK33z7freaRO_d"
DEST="data/kayuan" SPLIT_ROOT="data/kayuan_sequence" WORKERS=16 SKIP_SPLIT=0 API_KEY="${GDRIVE_API_KEY:-}"

usage() { sed -n '2,10p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder-id) FOLDER_ID="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --split-root) SPLIT_ROOT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;   # bắt buộc: train/images > 5500 file
    --skip-split) SKIP_SPLIT=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Tham số lạ: $1" >&2; usage 1 ;;
  esac
done

cd "$(dirname "$0")/../.."

# Key có thể để trong .env ở gốc repo (đã gitignore): GDRIVE_API_KEY=...
if [[ -z "$API_KEY" && -f .env ]]; then
  API_KEY="$(sed -n 's/^GDRIVE_API_KEY=//p' .env | tail -n 1)"
fi

# Chỉ lấy ảnh và nhãn của ba split; bỏ labels.cache, file tile, v.v.
python scripts/data/fetch_gdrive_folder.py "$FOLDER_ID" "$DEST" --workers "$WORKERS" ${API_KEY:+--api-key "$API_KEY"} \
  --include '^(train|val|test)/(images|labels)/'

echo "== Kiểm tra số file =="
for s in train val test; do
  img="$DEST/$s/images" lbl="$DEST/$s/labels"
  [[ -d "$img" && -d "$lbl" ]] || { echo "Thiếu $img hoặc $lbl" >&2; exit 1; }
  printf '%-5s images=%s labels=%s\n' "$s" "$(find "$img" -type f ! -name '*.part' | wc -l)" "$(find "$lbl" -name '*.txt' | wc -l)"
done

# Folder trên Drive không có data.yaml; check_split_leakage.py lấy `names` từ đây.
# Thứ tự 16 class giống data/kayuan_sequence/data.yaml cũ.
if [[ ! -f "$DEST/data.yaml" ]]; then
  cat > "$DEST/data.yaml" <<EOF
path: $(realpath "$DEST")
train: train/images
val: val/images
test: test/images
names: {0: tlr, 1: tly, 2: tlg, 3: tlra, 4: tlya, 5: tlga, 6: tsra, 7: tsya, 8: tsga, 9: trra, 10: trya, 11: trga, 12: tura, 13: tuya, 14: tuga, 15: tln}
EOF
  echo "Đã tạo $DEST/data.yaml (16 class)."
fi

[[ $SKIP_SPLIT -eq 1 ]] && exit 0
if [[ -e "$SPLIT_ROOT" ]]; then
  echo "$SPLIT_ROOT đã tồn tại, không tạo lại split." >&2
  exit 0
fi
python scripts/data/check_split_leakage.py "$DEST" --kind kayuan --exclude-missing-labels \
  --output-root "$SPLIT_ROOT" --report "$SPLIT_ROOT.audit.json"
echo "Xong. So với doc/kayuan_split_audit.json: 14127 train / 3036 val / 3035 test."
