#!/usr/bin/env python3
"""Count parameters for a YOLO .pt model and a PyTorch .pth checkpoint."""

import argparse
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PT_MODEL = BASE_DIR / "yolo11m.pt"
DEFAULT_PTH_MODEL = BASE_DIR / "tile_proposal_cnn_model.pth"


def register_custom_modules():
    try:
        import ultralytics.nn.tasks as tasks
        from custom_modules import CBAM, SE
    except ImportError:
        return

    tasks.CBAM = CBAM
    tasks.SE = SE


def parse_args():
    parser = argparse.ArgumentParser(description="Count parameters in a YOLO .pt model and a .pth checkpoint")
    parser.add_argument("--pt-model", default=str(DEFAULT_PT_MODEL), help="Path to the YOLO .pt file")
    parser.add_argument("--pth-model", default=str(DEFAULT_PTH_MODEL), help="Path to the PyTorch .pth file")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used when loading the .pth checkpoint (default: cpu)",
    )
    return parser.parse_args()


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def load_yolo_model(model_path):
    register_custom_modules()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required to load YOLO .pt models. Install it in the active environment first."
        ) from exc

    return YOLO(model_path).model


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema"):
            value = checkpoint.get(key)
            if hasattr(value, "state_dict"):
                return value.state_dict()
            if isinstance(value, dict):
                checkpoint = value
                break

    if isinstance(checkpoint, dict):
        cleaned = {}
        for key, value in checkpoint.items():
            cleaned[key.replace("module.", "", 1)] = value
        return cleaned

    raise TypeError(f"Unsupported checkpoint format in {path}")


def load_pth_model(model_path, device):
    from evaluation import TileProposalCNN

    model = TileProposalCNN().to(device)
    state_dict = load_checkpoint(model_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    yolo_model = load_yolo_model(args.pt_model)
    pth_model = load_pth_model(args.pth_model, device)

    yolo_total = count_parameters(yolo_model)
    pth_total = count_parameters(pth_model)

    print(f"YOLO .pt model: {args.pt_model}")
    print(f"  Parameters:           {yolo_total:,}")
    print()
    print(f"PyTorch .pth model: {args.pth_model}")
    print(f"  Parameters:           {pth_total:,}")


if __name__ == "__main__":
    main()