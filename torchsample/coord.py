import torch
import torch.nn.functional as F

from . import default


def unnormalize(coord, size, align_corners, clip=False):
    """Unnormalize normalized coordinates.

    Modified from PyTorch C source:
        https://github.com/pytorch/pytorch/blob/c52290bad18d23d7decd37b581307f8241c0e8c5/aten/src/ATen/native/GridSampler.h#L26

    Parameters
    ----------
    coord : torch.Tensor
        Values in range ``[-1, 1]`` to unnormalize.
    size : int or tuple
        Unnormalized side length in pixels.
        If tuple, order is ``(x, y, ...)`` to match ``coord``.
    align_corners : bool
        if ``True``, the corner pixels of the input and output tensors are
        aligned, and thus preserving the values at those pixels.
    clip : bool
        If ``True``, output will be clipped to range ``[0, size-1]``

    Returns
    -------
    torch.Tensor
        Unnormalized coordinates.
    """
    size = coord.new_tensor(size)
    size = size[(None,) * (coord.ndim - 1)]
    if align_corners:
        # unnormalize coord from [-1, 1] to [0, size - 1]
        unnorm = ((coord + 1) / 2) * (size - 1)
    else:
        # unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        unnorm = ((coord + 1) * size - 1) / 2

    if clip:
        unnorm = torch.clip(unnorm, 0, size - 1)

    return unnorm


def normalize(coord, size, align_corners):
    """Normalize unnormalized coordinates.

    Related
    -------
    unnormalize

    Parameters
    ----------
    coord : torch.Tensor
        Values in range ``[0, size-1]`` to normalize.
    size : int or tuple
        Unnormalized side length in pixels.
        If tuple, order is ``(x, y, ...)`` to match ``coord``.
    align_corners : bool
        if ``True``, the corner pixels of the input and output tensors are
        aligned, and thus preserving the values at those pixels.

    Returns
    -------
    torch.Tensor
        Normalized coordinates.
    """
    size = coord.new_tensor(size)
    size = size[(None,) * (coord.ndim - 1)]

    if align_corners:
        norm = (coord / (size - 1)) * 2 - 1
    else:
        norm = (coord * 2 + 1) / size - 1

    return norm


def rand(batch, sample, dims=2, dtype=None, device=None):
    """Generate random coordinates in range ``[-1, 1]``.

    Parameters
    ----------
    batch : int
    sample : int
    dims : int
        ``2`` for generating ``(x, y)`` coordinates.
        ``3`` for generating ``(x, y, z)`` coordinates.
        Defaults to ``2``.
    dtype : torch.dtype
        The desired data type of returned tensor.
    device : torch.device
        The desired device of returned tensor

    Returns
    -------
        (batch, sample, dims) random coordinates in range ``[-1, 1]``.
    """
    return 2 * torch.rand(batch, sample, dims, dtype=dtype, device=device) - 1


def randint(batch, sample, size, device=None, align_corners=default.align_corners):
    """Generate random pixel coordinates in range ``[-1, 1]``.

    Unlike ``rand``, the resulting coordinates will land exactly on pixels.
    Intended for sampling high resolution GT data during data loading.

    Due to numerical precision, it is suggested to use ``mode="nearest"``
    when sampling featuremaps of resolution ``size`` using these coordinates.

    Parameters
    ----------
    batch : int
    sample : int
    size : tuple
        Size of field to generate pixel coordinates for. i.e. ``(x, y, ...)``.
    device : torch.device
        The desired device of returned tensor
    align_corners : bool
        if ``True``, the corner pixels of the input and output tensors are
        aligned, and thus preserving the values at those pixels.

    Returns
    -------
        (batch, sample, dims) random coordinates in range ``[-1, 1]``.
    """
    coords = []
    for s in size:
        unnorm = torch.randint(s, size=(batch, sample), device=device)
        norm = normalize(unnorm, s, align_corners)
        coords.append(norm)
    coords = torch.stack(coords, dim=-1)
    return coords


def full(size, device=None, align_corners=default.align_corners):
    """Generate 2D or 3D coordinates to fully sample an image.

    Parameters
    ----------
    size : tuple
        Tuple of length 4 (2D) ``(n, c, h, w)`` or 5 (3D) ``(n, c, d, h, w)``.
        In either case, ``c`` doesn't matter and is just there so we can
        conveniently use ``output.shape``.
    device : torch.device
        The desired device of returned tensor
    align_corners : bool
        if ``True``, the corner pixels of the input and output tensors are
        aligned, and thus preserving the values at those pixels.

    Returns
    -------
    coords : torch.Tensor
        ``(n, h, w, 2)`` Normalized coordinates for sampling functions.
    """
    batch = size[0]
    theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device)
    theta = theta.repeat(batch, 1, 1)
    norm_coords = F.affine_grid(theta, size, align_corners=align_corners).to(device)

    return norm_coords


def full_like(tensor, *args, **kwargs):
    return full(tensor.shape, *args, **kwargs)
