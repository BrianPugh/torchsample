"""Sanity integration tests between various components.
"""

import jittor as jt
import pytest

import torchsample.jittor as ts


def assert_close(actual, expected, atol=1e-4):
    assert ((expected - actual).abs() < atol).all()


@pytest.fixture
def single_batch():
    return jt.rand(1, 10, 3, 15)


def test_full_sample2d(single_batch):
    expected = single_batch.permute(0, 2, 3, 1)

    coords = ts.coord.full_like(single_batch)
    sampled = ts.sample2d(coords, single_batch)

    assert sampled.shape == (1, 3, 15, 10)

    assert_close(expected, sampled, atol=1e-6)


def test_full_sample2d_pos(single_batch):
    expected = single_batch.permute(0, 2, 3, 1)
    # Identity encoder just tacks on the normalized coord.
    encoder = ts.encoding.Identity()

    coords = ts.coord.full_like(single_batch)
    sampled = ts.sample2d(coords, single_batch, encoder=encoder)

    assert sampled.shape == (1, 3, 15, 10 + 2)

    assert_close(expected, sampled[..., :10], atol=1e-6)
    assert_close(coords, sampled[..., 10:], atol=1e-6)


def test_randint_align_corners_true():
    align_corners = True
    batch = jt.rand(2, 3, 480, 320)
    coords = ts.coord.randint(2, 4096, (320, 480), align_corners=align_corners)
    coords = ts.coord.randint_like(4096, batch, align_corners=align_corners)
    sampled = ts.sample2d(coords, batch, mode="nearest", align_corners=align_corners)

    sampled_flat = sampled.reshape(-1, 3)
    batch_flat = batch.permute(0, 2, 3, 1).reshape(-1, 3)
    assert all((x[None] == batch_flat).all(-1).any() for x in sampled_flat)


def test_randint_align_corners_false():
    align_corners = False
    batch = jt.rand(2, 3, 480, 320)
    coords = ts.coord.randint(2, 4096, (320, 480), align_corners=align_corners)
    coords = ts.coord.randint_like(4096, batch, align_corners=align_corners)
    sampled = ts.sample2d(coords, batch, mode="nearest", align_corners=align_corners)

    sampled_flat = sampled.reshape(-1, 3)
    batch_flat = batch.permute(0, 2, 3, 1).reshape(-1, 3)
    assert all((x[None] == batch_flat).all(-1).any() for x in sampled_flat)
