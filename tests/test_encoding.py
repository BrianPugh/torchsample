import torch

import torchsample as ts


def test_nearest_pixel_pixel_perfect():
    """Test coordinates that are exactly at pixel locations.

    Relative offsets should be 0 at these locations.
    """
    coords = torch.tensor(
        [
            # Top Row
            [-1.0, -1.0],  # top-left pixel; no offset
            [0.0, -1.0],  # top-center pixel; no offset
            [1.0, -1.0],  # top-right pixel; no offset
            # More complicated pixel-perfect coords
            [-1.0 + 2.0 * (1 / (5 - 1)), -1],  # second pixel right
            [-1, -1.0 + 2.0 * (1 / (3 - 1))],  # second pixel down
        ]
    )[None]
    actual = ts.encoding.nearest_pixel(coords, (3, 5))

    assert actual.shape == (1, 5, 2 + 2)

    # Top Row
    # [-1, -1]: top-left pixel
    # These 3 coordinates should land perfectly on pixels,
    # so the relative offsets should be 0.
    assert torch.allclose(actual[0, :3, 2:], torch.tensor(0.0))

    # More complicated pixel-perfect coords
    # Second pixel right
    assert torch.allclose(actual[0, 3, 2:], torch.tensor(0.0))
    # Second pixel down
    assert torch.allclose(actual[0, 4, 2:], torch.tensor(0.0))


def test_nearest_pixel_halfway():
    """Test coordinates exactly halfway between pixels, and slightly past halfway."""
    eps = 1e-6
    coords = torch.tensor(
        [
            # Intermediate halfway
            [-1.0 + 1.0 * (1 / (5 - 1)), -1],  # halfway to second pixel right
            [-1, -1.0 + 1.0 * (1 / (3 - 1))],  # halfway to second pixel down
            # Just over halfway
            [-1.0 + 1.0 * (1 / (5 - 1)) + eps, -1],  # halfway to second pixel right
            [-1, -1.0 + 1.0 * (1 / (3 - 1)) + eps],  # halfway to second pixel down
        ]
    )[None]
    actual = ts.encoding.nearest_pixel(coords, (3, 5))

    assert actual.shape == (1, 4, 2 + 2)

    # Test the norm_coord portion
    assert torch.allclose(coords.float(), actual[..., :2])

    # Intermediate halfway
    # torch.round rounds down, so these will be negative.
    # halfway to second pixel right
    assert torch.allclose(actual[0, 0, 2:], 2.0 * torch.tensor([-0.5, 0.0]))
    # halfway to second pixel down
    assert torch.allclose(actual[0, 1, 2:], 2.0 * torch.tensor([0.0, -0.5]))

    # Just over halfway
    # just over halfway to second pixel right
    assert torch.allclose(actual[0, 2, 2:], 2.0 * torch.tensor([0.5, 0.0]))
    # just over halfway to second pixel down
    assert torch.allclose(actual[0, 3, 2:], 2.0 * torch.tensor([0.0, 0.5]))
