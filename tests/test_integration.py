"""Sanity integration tests between various components.
"""

import pytest
import torch

import torchsample as ts


@pytest.fixture
def single_batch():
    return torch.rand(1, 10, 3, 15)


def test_full_sample2d(single_batch):
    align_corners = True

    expected = single_batch.permute(0, 2, 3, 1)

    coords = ts.coord.full_like(single_batch, align_corners=align_corners)
    sampled = ts.sample2d(coords, single_batch, align_corners=align_corners)

    assert sampled.shape == (1, 3, 15, 10)

    assert torch.allclose(expected, sampled, atol=1e-6)


def test_full_sample2d_pos(single_batch):
    align_corners = True

    expected = single_batch.permute(0, 2, 3, 1)
    # Identity encoder just tacks on the normalized coord.
    encoder = ts.encoding.Identity()

    coords = ts.coord.full_like(single_batch, align_corners=align_corners)
    sampled = ts.sample2d(
        coords, single_batch, align_corners=align_corners, encoder=encoder
    )

    assert sampled.shape == (1, 3, 15, 10 + 2)

    assert torch.allclose(expected, sampled[..., :10], atol=1e-6)
    assert torch.allclose(coords, sampled[..., 10:], atol=1e-6)
