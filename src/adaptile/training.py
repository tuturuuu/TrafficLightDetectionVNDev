"""
Train detector (YOLO + attention, có/không tiling) và tile selector từ một file config.

Dùng bởi scripts/train_detector.py và scripts/train_selector.py; xem
configs/train/*.yaml để biết từng mục.
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from .dataset_tiling import ensure_tiled

REPO = Path(__file__).resolve().parents[2]

# (arch, attention) -> file kiến trúc. Tên không có "/" là config có sẵn của Ultralytics.
ARCHS = {
    ("yolov8m", "none"): "yolov8m.yaml",
    ("yolov8m", "cbam"): "configs/models/YOLOv8_CBAM.yaml",
    ("yolo26m", "none"): "yolo26m.yaml",
    ("yolo26m", "se"): "configs/models/yolo26m-modified.yaml",
}


def resolve_arch(model_cfg):
    """model.arch là tên (yolov8m, yolo26m) + model.attention, hoặc đường dẫn .yaml/.pt."""
    arch, att = model_cfg.arch, model_cfg.get("attention", "none") or "none"
    if arch.endswith((".yaml", ".pt")):
        return arch
    if (arch, att) not in ARCHS:
        have = ", ".join(f"{a}+{t}" for a, t in ARCHS)
        raise KeyError(f"Chưa có kiến trúc {arch}+{att}. Có sẵn: {have}. "
                       "Hoặc đặt model.arch là đường dẫn file .yaml")
    return ARCHS[(arch, att)]


def _label_dir(img_dir):
    """.../images/<split> -> .../labels/<split>; .../<split>/images -> .../<split>/labels"""
    parts = list(Path(img_dir).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts)


def prepare_detector_data(cfg):
    """Trả đường dẫn data yaml để train. tiling.type=uniform thì cắt tile (nếu
    chưa có) cho train/val và sinh <root>/data<suffix>.yaml."""
    data_yaml = Path(cfg.dataset.data_yaml)
    if cfg.tiling.type == "none":
        return str(data_yaml)
    if cfg.tiling.type != "uniform":
        raise KeyError(f"tiling.type '{cfg.tiling.type}' không hỗ trợ (uniform | none)")

    root = data_yaml.parent
    data = yaml.safe_load(data_yaml.read_text())
    t = cfg.tiling
    out = {}
    for split in ("train", "val"):
        img_dir = root / data[split]
        status = ensure_tiled(img_dir, _label_dir(img_dir), root / f"{split}{t.suffix}",
                              t.tile_size, t.overlap, t.min_visibility)
        print(f"{split}: {root / f'{split}{t.suffix}'} ({status})")
        out[split] = f"{split}{t.suffix}/images"
    out["names"] = data["names"]
    tiled_yaml = root / f"data{t.suffix}.yaml"
    tiled_yaml.write_text("# Sinh bởi scripts/train_detector.py\n"
                          + yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    return str(tiled_yaml)


def train_detector(cfg, dry_run=False, overwrite=False):
    from .detectors import register_custom_modules

    arch = resolve_arch(cfg.model)
    tr = cfg.train
    project = os.path.abspath(cfg.output.project)
    taken = [s for s in tr.seeds
             if os.path.exists(os.path.join(project, f"{cfg.output.name}_seed{s}"))]
    if taken and not overwrite:
        raise FileExistsError(f"Đã có run {cfg.output.name}_seed{taken} trong {project}; "
                              "đổi output.name hoặc thêm --overwrite")
    if dry_run:
        data = cfg.dataset.data_yaml if cfg.tiling.type == "none" else \
            f"(sẽ cắt tile và sinh {Path(cfg.dataset.data_yaml).parent}/data{cfg.tiling.suffix}.yaml)"
    else:
        data = prepare_detector_data(cfg)
    runs = []
    for seed in tr.seeds:
        name = f"{cfg.output.name}_seed{seed}"
        kwargs = dict(data=data, epochs=tr.epochs, imgsz=tr.imgsz, batch=tr.batch,
                      project=project, name=name, device=tr.device,
                      workers=tr.workers, patience=tr.patience, seed=seed,
                      deterministic=tr.deterministic, exist_ok=overwrite,
                      **(OmegaConf.to_container(tr.extra) if "extra" in tr else {}))
        print(f"\n=== {name}: {arch}"
              + (f" <- {cfg.model.pretrained}" if cfg.model.get("pretrained") else "")
              + f"\n{kwargs}")
        runs.append(os.path.join(project, name))
        if dry_run:
            continue
        register_custom_modules()
        from ultralytics import YOLO
        model = YOLO(arch)
        if cfg.model.get("pretrained"):
            model.load(cfg.model.pretrained)
        model.train(**kwargs)
    return runs


def train_selector(cfg, dry_run=False):
    kind = cfg.type
    if kind not in ("gridnet", "tile_cnn"):
        raise KeyError(f"type '{kind}' không hỗ trợ (gridnet | tile_cnn)")
    params = OmegaConf.to_container(cfg[kind])
    print(f"=== train {kind} -> {cfg.out}\n{params}")
    if dry_run:
        return
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    if kind == "gridnet":
        from .models.gridnet import train_gridnet
        train_gridnet(out=cfg.out, **params)
    else:
        # CNN thế hệ 1: legacy/cnn_classifier/small_cnn.py là script chạy từ trên xuống, gọi qua subprocess
        cmd = [sys.executable, str(REPO / "legacy/cnn_classifier/small_cnn.py"), "--out", cfg.out]
        for k, v in params.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        subprocess.run(cmd, check=True)
