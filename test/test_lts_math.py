import torch

from adaptile.models.lts import LTSHead, parameter_free_map
from adaptile.relevance_targets import relevance_loss


def test_exact_projection_and_parameter_count():
    head = LTSHead(3)
    assert sum(p.numel() for p in head.parameters()) == 4
    assert len(list(head.children())) == 1
    x = torch.randn(2, 3, 5, 7)
    result = head(x)
    assert result.shape == (2, 1, 5, 7)
    torch.testing.assert_close(result, torch.nn.functional.conv2d(x, head.proj.weight, head.proj.bias))
    loss = relevance_loss(result, torch.zeros_like(result), torch.ones_like(result), 3)
    loss.backward()
    assert head.proj.weight.grad.abs().sum() > 0


def test_sigmoid_is_before_channel_average():
    features = torch.tensor([[[[-10.]], [[2.]]]])
    expected = (torch.sigmoid(features[:, :1]) + torch.sigmoid(features[:, 1:])) / 2
    torch.testing.assert_close(parameter_free_map(features), expected)
    assert not torch.allclose(expected, torch.sigmoid(features.mean(1, keepdim=True)))


def test_empty_targets_and_padding_loss_are_finite():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss = relevance_loss(logits, torch.zeros_like(logits), torch.zeros_like(logits))
    assert loss == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
