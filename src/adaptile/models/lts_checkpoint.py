"""Load an LTS head only with its exact paired detector and feature geometry."""

from pathlib import Path

import torch

from .lts import LTSHead
from ..provenance import file_sha256

MODES = ('frozen_head_pilot', 'joint')
FEATURE_KEYS = ('architecture_sha256', 'feature_stage', 'channels', 'stride_hw',
                'coarse_imgsz', 'preprocessing')


def save_lts_checkpoint(path, head, provider, detector_path, training_mode, training_metadata,
                        overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if training_mode not in MODES:
        raise ValueError(f'training_mode must be one of {MODES}')
    metadata = {**provider.metadata, 'detector_sha256': file_sha256(detector_path),
                'training_mode': training_mode, 'training': training_metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': 'adaptile_lts_v1', 'metadata': metadata,
                'head': {k: v.detach().cpu() for k, v in head.state_dict().items()}}, path)
    return metadata


def load_lts_checkpoint(path, provider, detector_path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'adaptile_lts_v1':
        raise ValueError('Not an adaptile LTS checkpoint (GridNet/TileCNN are separate methods)')
    metadata = checkpoint['metadata']
    for key in FEATURE_KEYS:
        if metadata.get(key) != provider.metadata[key]:
            raise ValueError(f'Incompatible LTS feature metadata: {key}')
    if metadata.get('training_mode') not in MODES:
        raise ValueError('Missing or unknown LTS training mode')
    if metadata.get('detector_sha256') != file_sha256(detector_path):
        raise ValueError('LTS detector SHA256 mismatch; load the matched fine detector')
    if metadata.get('versions', {}).get('ultralytics') != provider.metadata['versions']['ultralytics']:
        raise ValueError('Ultralytics version mismatch; reverify feature compatibility first')
    parameter = next(provider.model.parameters())
    head = LTSHead(provider.channels).to(device=parameter.device, dtype=parameter.dtype)
    head.load_state_dict(checkpoint['head'], strict=True)
    head.eval()
    return head, metadata
