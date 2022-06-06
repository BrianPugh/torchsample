import jittor as jt

import torchsample.jittor as ts


def allclose(actual, expected):
    return ((expected - actual).abs() < 1e-5).all()


def test_gamma():
    """Only exercises code; doesn't assert correct results."""
    coords = jt.rand(3, 4, 2)
    actual = ts.encoding.gamma(coords)
    assert actual.shape == (3, 4, 40)


def test_nearest_pixel_pixel_perfect():
    """Test coordinates that are exactly at pixel locations.

    Relative offsets should be 0 at these locations.
    """
    coords = jt.float(
        [
            # Top Row
            [-0.8, -2.0 / 3],  # top-left pixel; no offset
            [0.0, -2.0 / 3],  # top-center pixel; no offset
            [0.8, -2.0 / 3],  # top-right pixel; no offset
        ]
    )[None]
    actual = ts.encoding.nearest_pixel(coords, (5, 3))

    assert actual.shape == (1, 3, 2 + 2)

    assert allclose(actual[0, :3, 2:], 0.0)


def test_nearest_pixel_halfway():
    """Test coordinates exactly halfway between pixels, and slightly past halfway."""
    eps = 1e-6
    coords = jt.float(
        [
            # Intermediate halfway
            [-0.8 + 0.2, -2.0 / 3],  # halfway to second pixel right
            [-0.8, -1.0 / 3],  # halfway to second pixel down
            # Intermediate halfway
            # Just over halfway
            [-0.8 + 0.2 + eps, -2.0 / 3],  # halfway to second pixel right
            [-0.8, -1.0 / 3 + eps],  # halfway to second pixel down
        ]
    )[None]
    actual = ts.encoding.nearest_pixel(coords, (5, 3))

    assert actual.shape == (1, 4, 2 + 2)

    # Test the norm_coord portion
    assert allclose(coords.float(), actual[..., :2])

    # Intermediate halfway
    # halfway to second pixel right
    assert allclose(actual[0, 0, 2:], 2.0 * jt.float([-0.5, 0.0]))
    # halfway to second pixel down
    assert allclose(actual[0, 1, 2:], 2.0 * jt.float([0.0, -0.5]))

    # Just over halfway
    # just over halfway to second pixel right
    assert allclose(actual[0, 2, 2:], 2.0 * jt.float([0.5, 0.0]))
    # just over halfway to second pixel down
    assert allclose(actual[0, 3, 2:], 2.0 * jt.float([0.0, 0.5]))
