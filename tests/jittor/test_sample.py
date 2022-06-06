import jittor as jt

import torchsample.jittor as ts


def assert_close(actual, expected):
    assert ((expected - actual).abs() < 1e-4).all()


def test_unsqueeze_at():
    unsqueeze_at = ts._sample._unsqueeze_at
    tensor = jt.rand(10, 3, 480, 640)

    assert unsqueeze_at(tensor, 0, 0).shape == (10, 3, 480, 640)
    assert unsqueeze_at(tensor, 0, 1).shape == (1, 10, 3, 480, 640)
    assert unsqueeze_at(tensor, 0, 2).shape == (1, 1, 10, 3, 480, 640)

    assert unsqueeze_at(tensor, 2, 0).shape == (10, 3, 480, 640)
    assert unsqueeze_at(tensor, 2, 1).shape == (10, 3, 1, 480, 640)
    assert unsqueeze_at(tensor, 2, 2).shape == (10, 3, 1, 1, 480, 640)


def test_squeeze_at():
    squeeze_at = ts._sample._squeeze_at
    tensor = jt.rand(10, 3, 1, 1, 480, 640)

    assert squeeze_at(tensor, 0, 0).shape == (10, 3, 1, 1, 480, 640)
    assert squeeze_at(tensor, 2, 1).shape == (10, 3, 1, 480, 640)
    assert squeeze_at(tensor, 2, 2).shape == (10, 3, 480, 640)


def test_sample2d_coords_shape3():
    coords = jt.float(
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
            # Any value below -0.5 shouldn't change value.
            [-0.9, -1],
            [-0.6, -1],
            [-0.5, -1],
            # Coordinates that are a little less nice
            [-0.5 + 0.25, -0.5],  # 1/4 pixel towards (1x, 0y)
            [-0.5, -0.5 + 0.25],  # 1/4 pixel towards (0x, 1y)
        ]
    )[None]
    n_coords = coords.shape[1]

    featmap = jt.float([[10.0, 20.0], [30.0, 40.0]])[None, None]
    featmap = featmap.repeat(1, 5, 1, 1)

    actual = ts.sample2d(coords, featmap, encoder=ts.encoding.identity)
    assert actual.shape == (1, n_coords, 7)

    actual = ts.sample2d(coords, featmap)
    assert actual.shape == (1, n_coords, 5)

    # Top Row
    # [-1, -1]: top-left pixel
    # This in unnormalized coordinates gets mapped to (-0.5, -0.5).
    assert_close(actual[0, 0], 10.0)
    # [0, -1]: middle between 10 and 20
    assert_close(actual[0, 1], 15.0)
    # [1, -1]: top-right pixel
    assert_close(actual[0, 2], 20.0)

    # Bottom Row
    # [-1, 1]: bottom-left pixel
    assert_close(actual[0, 3], 30.0)
    # [0, 1]: middle between 30 and 40
    assert_close(actual[0, 4], 35.0)
    # [0, 1]: bottom-right pixel
    assert_close(actual[0, 5], 40.0)

    # Trying out small interpolations.
    # These shouldn't change value due to border padding.
    assert_close(actual[0, 6], 10.0)
    assert_close(actual[0, 7], 10.0)
    assert_close(actual[0, 8], 10.0)

    # Coordinates that are a little less nice
    assert_close(actual[0, 9], 12.5)
    assert_close(actual[0, 10], 15.0)


def test_sample_unified_2d():
    featmap = jt.rand(10, 3, 192, 256)
    coords = ts.coord.rand(10, 4096)

    sample_out = ts.sample(coords, featmap)
    sample2d_out = ts.sample2d(coords, featmap)
    assert_close(sample_out, sample2d_out)


def test_sample3d_coords_shape4():
    # TODO
    pass


def test_sample_unified_3d():
    featmap = jt.rand(10, 3, 5, 192, 256)
    coords = ts.coord.rand(10, 4096, 3)

    sample_out = ts.sample(coords, featmap)
    sample3d_out = ts.sample3d(coords, featmap)
    assert_close(sample_out, sample3d_out)
