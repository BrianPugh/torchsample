import pytest
import torch

import torchsample as ts


def test_rand():
    actual = ts.coord.rand(5, 4096)
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (5, 4096, 2)
    assert 0.99 < actual.max() <= 1.0
    assert -0.99 > actual.min() >= -1.0


def test_full_single_batch_2d():
    h, w = 2, 3
    coords = ts.coord.full((1, 1, h, w), align_corners=True)
    assert coords.shape == (1, h, w, 2)

    # Verify coords are in correct order (xy).
    assert (coords[0, 0, 0] == torch.Tensor([-1.0, -1.0])).all()

    # Moving down an image should increase y
    assert (coords[0, 1, 0] == torch.Tensor([-1.0, 1.0])).all()

    # Moving ot the right should increase x
    assert (coords[0, 0, 1] == torch.Tensor([0, -1.0])).all()


def test_full_like_match_full():
    tensor = torch.rand(3, 4, 5, 6)
    full_coords = ts.coord.full(tensor.shape)
    full_like_coords = ts.coord.full_like(tensor)

    assert full_coords.shape == full_like_coords.shape
    assert full_like_coords.shape == (3, 5, 6, 2)
    assert torch.allclose(full_coords, full_like_coords)
