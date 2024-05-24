"""Unit testing for the modeling script."""


import torch

from houselec import modeling


def test_forecaster():
    """Test if model output has the correct size."""
    X = torch.randn((2, 3, 4))
    model = modeling.Forecaster(16, 8, 4, 2)
    model.eval()
    y = model(X)
    assert y.size() == torch.Size([2, 2])
