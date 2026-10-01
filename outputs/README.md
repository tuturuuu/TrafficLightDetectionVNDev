# outputs/

Kết quả thí nghiệm, chia theo dataset: `outputs/<dataset>/...`. Script mặc định ghi vào `outputs/` (vd `dense_tiling_experiment.py --out outputs/<dataset>/scaling/<tên_run>.json`).

> ⚠️ Các số liệu ở đây được tính bằng `evaluate_accuracy` **chưa xét class**, và split Kayuan bị rò rỉ dữ liệu train/test. Xem mục "Lưu ý quan trọng" trong README gốc trước khi dùng số cho paper.

## kayuan/

| File / thư mục | Nội dung |
|---|---|
| `scaling_results_20260817.json` | Kết quả `dense_tiling_experiment.py` đời đầu (chỉ tile 640). Dataset suy ra từ số tile/ảnh, chưa xác nhận. Tên cũ `scaling_results.json` |
| `kayuan_seed0_grid_test.json` | Kết quả `dense_tiling_experiment.py`, seed 0 |
| `benchmark_results/` | Ảnh heatmap GridNet (`grid_proposal_net.py benchmark`) |
| `gradcam_out/` | Grad-CAM của CNN chọn tile thế hệ 1 |

Kết quả chính của Kayuan (3 seed, uniform với adaptive) hiện vẫn nằm ở `eval_logs/kayuan/adaptive/gridnet/`.

## seadronesee/

| File / thư mục | Nội dung |
|---|---|
| `scaling/scaling_test_seed{0,42,100}.json` | Uniform với adaptive trên test, tile 640, 3 seed |
| `scaling/scaling_test_tilesize_sweep_seed0.json` | Sweep tile size 640 / 800 / 960 |
| `scaling/scaling_val_pw8_thr{0.03,0.05,0.08}_seed0.json` | Sweep threshold GridNet trên val (`pw8` = pos-weight 8) |
| `scaling/scaling_results_seed0.json` | Lần chạy đầu (2026-08-23) |
| `report_metrics.{md,csv}` | YOLOv8-CBAM **có tiling**, 3 seed (`collect_report_metrics.py`) |
| `report_metrics_notiled.{md,csv}` | YOLOv8-CBAM **không tiling**, 3 seed |
| `benchmark_results_pw8_thr*/` | Ảnh heatmap GridNet theo threshold (gitignored) |
