import numpy as np
import pytest
import torch

from adaptile.detectors import register_custom_modules
from adaptile.models.coarse_features import YOLOCoarseFeatureProvider, letterbox


@pytest.fixture
def model():
    from ultralytics import YOLO
    torch.set_num_threads(2)
    register_custom_modules()
    return YOLO('configs/models/yolo26m-p2.yaml').model.eval()


def test_features_match_reference_and_skip_detection_head(model):
    provider = YOLOCoarseFeatureProvider(model, 4, 64)
    image = np.random.default_rng(0).integers(0, 256, (35, 111, 3), dtype=np.uint8)
    array, _ = letterbox(image, 64)
    captured = []
    hook = model.model[4].register_forward_hook(lambda m, i, o: captured.append(o.detach().clone()))
    with torch.no_grad():
        model(torch.from_numpy(array.copy()).unsqueeze(0).float() / 255)
    hook.remove()
    head = model.model[-1].register_forward_pre_hook(lambda m, i: pytest.fail('Detection head ran'))
    result = provider(image)
    head.remove()
    torch.testing.assert_close(result.features, captured[0])
    assert provider.model is model
    assert result.features.shape == (1, 512, 8, 8)
    assert result.geometries[0].original_hw == (35, 111)


def test_batch_and_repeated_calls_and_gradients(model):
    provider = YOLOCoarseFeatureProvider(model, 2, 64)
    a = np.zeros((75, 31, 3), np.uint8)
    b = np.full((27, 103, 3), 200, np.uint8)
    first = provider(a).features.clone()
    combined = provider([a, b]).features
    torch.testing.assert_close(first, combined[:1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(first, provider(a).features)
    provider([a, b], grad_enabled=True).features.square().mean().backward()
    assert model.model[0].conv.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.model[-1].parameters())


@pytest.mark.parametrize('definition,stage', [('yolo26m-se', 5), ('yolo26m-cbam', 5)])
def test_attention_feature_path(definition, stage):
    from ultralytics import YOLO
    register_custom_modules()
    model = YOLO(f'configs/models/{definition}.yaml').model.eval()
    provider = YOLOCoarseFeatureProvider(model, stage, 64)
    assert provider(np.zeros((99, 35, 3), np.uint8)).features.shape == (1, 512, 8, 8)


def test_invalid_stage_and_letterbox_pixels(model):
    from ultralytics.data.augment import LetterBox
    for shape in [(35, 111), (101, 27), (64, 64)]:
        image = np.ones((*shape, 3), np.uint8)
        ours, geometry = letterbox(image, 64)
        reference = LetterBox(new_shape=(64, 64), auto=False)(image=image)
        np.testing.assert_array_equal(ours.transpose(1, 2, 0)[:, :, ::-1], reference)
        assert geometry.input_hw == (64, 64)
    with pytest.raises(ValueError, match='backbone'):
        YOLOCoarseFeatureProvider(model, len(model.model) - 1, 64)
