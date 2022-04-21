import pytest
import torch
from torch.testing import assert_close

import torchsample as ts


def test_normalize():
    unnorm_x = torch.tensor([0, 480 - 1])
    actual = ts.coord.normalize(unnorm_x, 480, align_corners=True)
    assert_close(actual, torch.tensor([-1.0, 1.0]))
    actual = ts.coord.normalize(unnorm_x, 480, align_corners=False)
    assert_close(actual, torch.tensor([-0.99792, 0.99792]))


def test_normalize_tuple():
    unnorm = torch.tensor(
        [
            [0, 0],
            [640 - 1, 480 - 1],
        ]
    )
    actual = ts.coord.normalize(unnorm, (640, 480), align_corners=True)
    assert_close(
        actual,
        torch.tensor(
            [
                [-1.0, -1.0],
                [1.0, 1.0],
            ]
        ),
    )
    actual = ts.coord.normalize(unnorm, (640, 480), align_corners=False)
    assert_close(
        actual,
        torch.tensor(
            [
                [-0.99844, -0.99792],
                [0.99844, 0.99792],
            ]
        ),
    )


def test_unnormalize():
    norm_x = torch.tensor([-1, 1])
    actual = ts.coord.unnormalize(norm_x, 480, align_corners=True)
    assert_close(actual, torch.tensor([0.0, 480 - 1]))
    actual = ts.coord.unnormalize(norm_x, 480, align_corners=False)
    assert_close(actual, torch.tensor([-0.5, 480 - 0.5]))


def test_unnormalize_tuple():
    norm = torch.tensor(
        [
            [-1, -1],
            [1, 1],
        ]
    )
    actual = ts.coord.unnormalize(norm, (640, 480), align_corners=True)
    assert_close(
        actual,
        torch.tensor(
            [
                [0.0, 0.0],
                [640 - 1, 480 - 1],
            ]
        ),
    )
    actual = ts.coord.unnormalize(norm, (640, 480), align_corners=False)
    assert_close(
        actual,
        torch.tensor(
            [
                [-0.5, -0.5],
                [640 - 0.5, 480 - 0.5],
            ]
        ),
    )


def test_normalize_unnormalize_auto_align_corners_true():
    align_corners = True
    normalized = 2 * torch.rand(10, 4096, 2) - 1
    unnorm = ts.coord.unnormalize(normalized, 1024, align_corners=align_corners)
    actual = ts.coord.normalize(unnorm, 1024, align_corners=align_corners)
    assert_close(normalized, actual)


def test_normalize_unnormalize_auto_align_corners_false():
    align_corners = False
    normalized = 2 * torch.rand(10, 4096, 2) - 1
    unnorm = ts.coord.unnormalize(normalized, 1024, align_corners=align_corners)
    actual = ts.coord.normalize(unnorm, 1024, align_corners=align_corners)
    assert_close(normalized, actual)


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

    # Assert the final coord is [1., 1.]
    assert (coords[0, -1, -1] == torch.Tensor([1.0, 1.0])).all()


def test_full_like_match_full():
    tensor = torch.rand(3, 4, 5, 6)
    full_coords = ts.coord.full(tensor.shape)
    full_like_coords = ts.coord.full_like(tensor)

    assert full_coords.shape == full_like_coords.shape
    assert full_like_coords.shape == (3, 5, 6, 2)
    assert_close(full_coords, full_like_coords)


def test_rand_biased():
    """Only exercises the code. No assertions.

    Coordinates can be visualized by uncommenting the code in test.
    """
    pred = torch.tensor(
        [
            [
                [0.8, 0.5, 0.95],
                [0.2, 0.5, 0.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [0.2, 0.5, 0.05],
                [0.8, 0.5, 1.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )[None]

    actual = ts.coord.rand_biased(1, 1000, pred)

    if False:
        import matplotlib.pyplot as plt

        actual = actual[0].numpy()
        plt.scatter(actual[:, 0], actual[:, 1])
        plt.axis([-1, 1, -1, 1])
        plt.gca().invert_yaxis()
        plt.show()
