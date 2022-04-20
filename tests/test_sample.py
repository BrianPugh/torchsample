import torch
from torch.testing import assert_close

import torchsample as ts


def test_sample2d_coords_shape3():
    coords = torch.tensor(
        [
            # Top Row
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            # Bottom Row
            [-1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            # Trying out small interpolations.
            [-0.9, -1],
        ]
    )[None]
    n_coords = coords.shape[1]

    featmap = torch.tensor([[10.0, 20.0], [30.0, 40.0]])[None, None]
    featmap = featmap.repeat(1, 5, 1, 1)

    actual = ts.sample2d(coords, featmap)
    assert actual.shape == (1, n_coords, 5)

    # Top Row
    # [-1, -1]: top-left pixel
    # This in unnormalized coorinates gets mapped to (-0.5, -0.5).
    # If ``align_corners=True, padding_mode="zeros"``,
    # then the interpolated value is ``2.5``.
    assert torch.allclose(actual[0, 0], torch.tensor(10.0))
    # [0, -1]: middle between 10 and 20
    assert torch.allclose(actual[0, 1], torch.tensor(15.0))
    # [1, -1]: top-right pixel
    assert torch.allclose(actual[0, 2], torch.tensor(20.0))

    # Bottom Row
    # [-1, 1]: bottom-left pixel
    assert torch.allclose(actual[0, 3], torch.tensor(30.0))
    # [0, 1]: middle between 30 and 40
    assert torch.allclose(actual[0, 4], torch.tensor(35.0))
    # [0, 1]: bottom-right pixel
    assert torch.allclose(actual[0, 5], torch.tensor(40.0))

    # Trying out small interpolations.
    assert torch.allclose(actual[0, 6], torch.tensor(0.95 * 10 + 0.05 * 20))


def test_sample_unified_2d():
    featmap = torch.rand(10, 3, 192, 256)
    coords = ts.coord.rand(10, 4096)

    sample_out = ts.sample(coords, featmap)
    sample2d_out = ts.sample2d(coords, featmap)
    assert_close(sample_out, sample2d_out)
