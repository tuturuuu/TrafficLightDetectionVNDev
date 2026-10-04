# outputs/

Kết quả thí nghiệm, chia theo dataset: `outputs/<dataset>/...`. `scripts/evaluate.py` ghi vào đường dẫn `eval.out` trong config (vd `outputs/kayuan/gridnet_scaling_seed0.json`), kèm file `.yaml` cùng tên ghi lại config đã chạy.

> ⚠️ Các file `scaling*.json` cũ (từ `legacy/dense_tiling_experiment.py`) được tính bằng `evaluate_accuracy` **chưa xét class**, và split Kayuan bị rò rỉ dữ liệu train/test. Xem mục "Lưu ý quan trọng" trong README gốc trước khi dùng số cho paper.

## kayuan/

| File / thư mục | Nội dung |
|---|---|
| `scaling_results_20260817.json` | Kết quả `legacy/dense_tiling_experiment.py` đời đầu (chỉ tile 640). Dataset suy ra từ số tile/ảnh, chưa xác nhận. Tên cũ `scaling_results.json` |
| `kayuan_seed0_grid_test.json` | Kết quả `legacy/dense_tiling_experiment.py`, seed 0 |
| `benchmark_results/` | Ảnh heatmap GridNet (`scripts/benchmark_gridnet.py benchmark`) |
| `gradcam_out/` | Grad-CAM của CNN chọn tile thế hệ 1 |

Kết quả GridNet 3 seed trên Kayuan (trước nằm ở `eval_logs/`) đã bị xoá; chạy lại bằng `python scripts/evaluate.py configs/eval/kayuan.yaml` với từng seed.

## seadronesee/

| File / thư mục | Nội dung |
|---|---|
| `scaling/scaling_test_seed{0,42,100}.json` | Uniform với adaptive trên test, tile 640, 3 seed. Seed 0 dùng GridNet `gridnet_seadronesee_8x8.pth`, threshold **0.03** (suy ra từ số tile giữ lại 14.06/ảnh) |
| `scaling/scaling_test_tilesize_sweep_seed0.json` | Sweep tile size 640 / 800 / 960. GridNet `gridnet_seadronesee_8x8.pth`, threshold **0.10** (suy ra từ 12.2 tile/ảnh ở tile 640) |
| `scaling/scaling_val_pw8_thr{0.03,0.05,0.08}_seed0.json` | Sweep threshold GridNet trên val (`pw8` = pos-weight 8) |
| `scaling/scaling_results_seed0.json` | Lần chạy đầu (2026-08-23) |
| `report_metrics.{md,csv}` | YOLOv8-CBAM **có tiling**, 3 seed (`scripts/collect_report_metrics.py`) |
| `report_metrics_notiled.{md,csv}` | YOLOv8-CBAM **không tiling**, 3 seed |
| `benchmark_results_pw8_thr*/` | Ảnh heatmap GridNet theo threshold (gitignored) |
