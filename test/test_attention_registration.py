import io

import pytest
import torch

from adaptile.models.attention import CBAM, SE


@pytest.mark.parametrize('kind', [CBAM, SE])
@pytest.mark.parametrize('channels', [8, 32])
def test_explicit_registration_dtype_gradients_and_state_roundtrip(kind, channels):
    module = kind().materialize(channels, dtype=torch.float64)
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    x = torch.randn(2, channels, 5, 7, dtype=torch.float64)
    module(x).square().mean().backward()
    assert all(id(p) in ids and p.dtype == x.dtype for p in module.parameters())
    assert all(p.grad is not None for p in module.parameters())
    assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0
    optimizer.step()
    buffer = io.BytesIO(); torch.save(module.state_dict(), buffer); buffer.seek(0)
    restored = kind().materialize(channels, dtype=torch.float64)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(module(x), restored(x))


@pytest.mark.parametrize('attention', ['se', 'cbam'])
def test_yolo26_attention_registered_before_optimizer(attention):
    from ultralytics import YOLO
    from adaptile.detectors import register_custom_modules
    register_custom_modules(); torch.set_num_threads(2)
    model = YOLO(f'configs/models/yolo26m-{attention}.yaml').model
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    expected = {id(p) for g in optimizer.param_groups for p in g['params']}
    params = sum(p.numel() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 64, 64))
    assert params == sum(p.numel() for p in model.parameters())
    assert expected == {id(p) for p in model.parameters()}
    assert model.stride.tolist() == [4, 8, 16, 32]
