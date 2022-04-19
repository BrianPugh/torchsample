import torch
import torch.nn.functional as F

from torchsample import default


def unnormalize(coord, size, align_corners, clip=True):
    """Unnormalize normalized coordinates.

    Modified from PyTorch C source:
        https://github.com/pytorch/pytorch/blob/c52290bad18d23d7decd37b581307f8241c0e8c5/aten/src/ATen/native/GridSampler.h#L26

    Parameters
    ----------
    coord : torch.Tensor
        Values in range ``[-1. 1]`` to unnormalize.
    size : int
        Unnormalized side length in pixels.
    align_corners : bool
    clip : bool
        If ``True``, output will be clipped to range ``[0, size-1]``

    Returns
    -------
    torch.Tensor
        Unnormalized coordinates.
    """
    if align_corners:
        # unnormalize coord from [-1, 1] to [0, size - 1]
        unnorm = ((coord + 1) / 2) * (size - 1)
    else:
        # unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        unnorm = ((coord + 1) * size - 1) / 2

    if clip:
        unnorm = torch.clip(unnorm, 0, size - 1)

    return unnorm


def rand(batch, sample, dims=2, dtype=None, device=None):
    """Generate random coordinates in range ``[-1, 1]``.

    Parameters
    ----------
    batch : int
    sample : int
    dims : 2
        ``2`` for generating ``(x, y)`` coordinates.
        ``3`` for generating ``(x, y, z)`` coordinates.
    dtype : torch.dtype
        The desired data type of returned tensor.
    device : torch.device
        The desired device of returned tensor

    Returns
    -------
        (batch, sample, dims) random coordinates in range ``[-1, 1]``.
    """
    return 2 * torch.rand(batch, sample, dims, dtype=dtype, device=device) - 1


def full(size, dtype=None, device=None, align_corners=default.align_corners):
    """Generate 2D or 3D coordinates to fully sample an image.

    Parameters
    ----------
    size : tuple
        Tuple of length 4 (2D) ``(n, c, h, w)`` or 5 (3D) ``(n, c, d, h, w)``.
        In either case, ``c`` doesn't matter and is just there so we can
        conveniently use ``output.shape``.

    Returns
    -------
    coords : torch.Tensor
        ``(n, h, w, 2)`` Normalized coordinates for sampling functions.
    """
    batch = size[0]
    theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device)
    theta = theta.repeat(batch, 1, 1)
    norm_coords = F.affine_grid(theta, size, align_corners=align_corners).to(device)

    # mesh_tensors = [torch.arange(x) for x in size[2:]]
    # index_coords = torch.stack(torch.meshgrid(*mesh_tensors, indexing="ij"), -1)[None]
    # Reshape them to be (n, samples, 2 or 3)
    # index_coords = index_coords.reshape(batch, -1, n_dim)
    # norm_coords = norm_coords.reshape(batch, -1, n_dim)
    # return index_coords, norm_coords
    return norm_coords


def full_like(tensor, *args, **kwargs):
    return full(tensor.shape, *args, **kwargs)
