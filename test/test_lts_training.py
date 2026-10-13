import numpy as np
import pytest
import torch

from adaptile.detectors import YOLODetector, register_custom_modules
from adaptile.selectors import LTSSelector
from adaptile.training_lts import LTSSystem, export_pair, fine_training_batch


def make_system(mode):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    register_custom_modules()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    detector = YOLO('configs/models/yolo26m-p2.yaml').model
    detector.args = get_cfg(overrides={'epochs': 2})
    return LTSSystem(detector, 4, 64, mode, fine_imgsz=64).train()


def samples():
    rng = np.random.default_rng(0)
    return [{'image': rng.integers(0, 255, (81, 99, 3), dtype=np.uint8),
             'boxes': [(12, 12, 28, 32, 0)], 'tile': (0, 0, 64, 64, 0, 0)} for _ in range(2)]


def norm(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


def test_joint_losses_reach_intended_parameters_and_update_once():
    system = make_system('joint')
    values = system.losses(samples())
    values['rel_loss'].backward(retain_graph=True)
    assert norm(system.head.parameters()) > 0
    assert norm(system.detector.model[0].parameters()) > 0
    assert norm(system.detector.model[-1].parameters()) == 0
    system.zero_grad(set_to_none=True)
    values['det_loss'].backward()
    assert norm(system.head.parameters()) == 0
    assert norm(system.detector.model[0].parameters()) > 0
    assert norm(system.detector.model[-1].parameters()) > 0
    optimizer = torch.optim.SGD(system.parameters(), lr=.001)
    before_head = system.head.proj.weight.detach().clone()
    before_backbone = system.detector.model[0].conv.weight.detach().clone()
    optimizer.zero_grad(); system.losses(samples())['loss'].backward(); optimizer.step()
    assert not torch.equal(before_head, system.head.proj.weight)
    assert not torch.equal(before_backbone, system.detector.model[0].conv.weight)


def test_frozen_pilot_and_matched_reload(tmp_path):
    system = make_system('frozen_head_pilot')
    before = {k: v.clone() for k, v in system.detector.state_dict().items()}
    optimizer = torch.optim.SGD(system.head.parameters(), lr=.001)
    losses = system.losses(samples())
    assert losses['det_loss'] == 0
    losses['loss'].backward(); optimizer.step()
    assert norm(system.detector.parameters()) == 0
    for key, value in before.items():
        torch.testing.assert_close(system.detector.state_dict()[key], value, rtol=0, atol=0)
    path = export_pair(system, tmp_path / 'pair', {'split_manifest_sha256': 'synthetic'})
    detector = YOLODetector(path, device='cpu', imgsz=64)
    selector = LTSSelector(path.parent / 'lts.pt', feature_stage=4, coarse_imgsz=64)
    selector.bind_detector(detector)
    assert selector.checkpoint_metadata['training_mode'] == 'frozen_head_pilot'
    assert selector.provider.model is detector.model.model
    assert selector.select(samples()[0]['image'], [(0, 0, 64, 64, 0, 0)]).shape == (1,)


def test_fine_targets_are_transformed_with_corresponding_crop():
    batch = fine_training_batch(samples(), 64, 'cpu')
    assert batch['batch_idx'].tolist() == [0, 1]
    torch.testing.assert_close(batch['bboxes'][0], torch.tensor([20/64, 22/64, 16/64, 20/64]))
