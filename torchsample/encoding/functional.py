import torch

from torchsample import default

from ..coord import unnormalize


def identity(coords):
    """Return ``coords`` unmodified."""
    return coords


def gamma(coords, order=10):
    """Positional encoding via sin and cos.

    From:
        NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

    Parameters
    ----------
    coords : torch.Tensor
        ``(..., dim)`` Coordinates to convert to positional encoding.
        In range ``[-1, 1]``.
    order : int
        Number

    Returns
    -------
    torch.Tensor
        ``(..., 2*dim*order)``
    """
    output = []
    for o in range(order):
        freq = (2**o) * torch.pi
        cos = torch.cos(freq * coords)
        sin = torch.sin(freq * coords)
        output.append(cos)
        output.append(sin)

    output = torch.cat(output, dim=-1)

    return output


def nearest_pixel(coords, size, align_corners=default.align_corners):
    """Encode normalized coords and relative offset to nearest neighbor.

    Note: offsets are multiplied by 2, so that their range is ``[-1, 1]``
    instead of ``[-0.5, 0.5]``

    From:
        Learning Continuous Image Representation with Local Implicit Image Function

    Parameters
    ----------
    coords : torch.Tensor
        ``(b, h, w, 2)`` Coordinates to convert to positional encoding.
        In range ``[-1, 1]``.
    size : tuple
        Size of field to generate pixel-center offsets for. i.e. ``(x, y, ...)``.

    Returns
    -------
    torch.Tensor
        ``(..., 4*dim)``
    """
    unnorm_coords = unnormalize(coords, size, align_corners)
    # 2x to scale the range from [-0.5, 0.5] to [-1, 1]
    # Note: torch rounds 0.5 DOWN
    unnorm_offset = 2 * (torch.round(unnorm_coords) - unnorm_coords)
    output = torch.cat((coords, unnorm_offset), dim=-1)
    return output
