import torch
import torch.nn.functional as F

from . import default


def sample2d(
    coords,
    featmap,
    mode="bilinear",
    padding_mode="border",
    align_corners=default.align_corners,
    encoder=None,
):
    """Sample a featmap at specified normalized coords.

    Parameters
    ----------
    coords : torch.Tensor
        ``(b, samples, dim)`` or ``(b, h, w, dim)`` coordinates in range
        ``[-1, 1]`` where the last dimension represents ``(x, y)``
        or ``(x, y, z)``.
    featmap : torch.Tensor
        ``(b, c, h, w)`` 2D featuremap.
    encoder : callable
        Converts coords to positional encoding.
        If provided, concatenates the positional encoding to the
        end of the returned samples.
    mode : str
        Interpolation method.
    padding_mode : str
        Defaults to ``"border"`` (different from pytorch default ``"zeros"``).

    Returns
    -------
    torch.Tensor
        ``(b, samples, c)`` or ``(b, h, w, c)``. Sampled featuremap.
        Features are last dimension.
    """
    remove_singleton = False
    if coords.ndim == 3:
        remove_singleton = True
        coords = coords[:, None]
    elif coords.ndim == 4:
        pass
    else:
        raise ValueError(f"Unknown coords shape {coords.shape=}.")

    # coords are 4D at this point.

    output = F.grid_sample(
        featmap,
        coords,
        padding_mode=padding_mode,
        mode=mode,
        align_corners=align_corners,
    )

    if encoder is not None:
        encoded = encoder(coords.permute(0, 3, 1, 2))
        output = torch.cat([output, encoded], 1)

    output = output.permute(0, 2, 3, 1)

    if remove_singleton:
        output = output[:, 0]

    return output


def sample3d(
    coords,
    featmap,
    mode="bilinear",
    padding_mode="border",
    align_corners=default.align_corners,
    encoder=None,
):
    """Sample a featmap at specified normalized coords.

    Parameters
    ----------
    coords : torch.Tensor
        ``(b, samples, dim)`` or ``(b, h, w, dim)`` coordinates in range
        ``[-1, 1]`` where the last dimension represents ``(x, y)``
        or ``(x, y, z)``.
    featmap : torch.Tensor
        ``(b, c, d, h, w)`` 3D featuremap.
    encoder : callable
        Converts coords to positional encoding.
        If provided, concatenates the positional encoding to the
        end of the returned samples.
    mode : str
        Interpolation method.
    padding_mode : str
        Defaults to ``"border"`` (different from pytorch default ``"zeros"``).

    Returns
    -------
    torch.Tensor
        ``(b, samples, c)`` or ``(b, d, h, w, c)``. Sampled featuremap.
        Features are last dimension.
    """
    remove_singleton = False
    if coords.ndim == 3:
        remove_singleton = True
        coords = coords[:, None, None]
    elif coords.ndim == 5:
        pass
    else:
        raise ValueError(f"Unknown coords shape {coords.shape=}.")

    # coords are 5D at this point.

    output = F.grid_sample(
        featmap,
        coords,
        padding_mode=padding_mode,
        mode=mode,
        align_corners=align_corners,
    )

    if encoder is not None:
        encoded = encoder(coords.permute(0, 4, 1, 2, 3))
        output = torch.cat([output, encoded], 1)

    output = output.permute(0, 2, 3, 4, 1)

    if remove_singleton:
        output = output[:, 0, 0]

    return output


def sample(
    coords,
    featmap,
    mode="bilinear",
    padding_mode="border",
    align_corners=default.align_corners,
    encoder=None,
):
    """Sample a featmap at specified normalized coords.

    Unified interface for both 2D and 3D operations.

    Parameters
    ----------
    coords : torch.Tensor
        ``(b, samples, dim)`` or ``(b, h, w, dim)`` coordinates in range
        ``[-1, 1]`` where the last dimension represents ``(x, y)``
        or ``(x, y, z)``.
    featmap : torch.Tensor
        ``(b, c, h, w)`` (2D) or ``(b, c, d, h, w)`` (3D).
    encoder : callable
        Converts coords to positional encoding.
        If provided, concatenates the positional encoding to the
        end of the returned samples.
    mode : str
        Interpolation method.
    padding_mode : str
        Defaults to ``"border"`` (different from pytorch default ``"zeros"``).

    Returns
    -------
    torch.Tensor
        ``(b, samples, c)`` or ``(b, h, w, c)``. Sampled featuremap.
        Features are last dimension.
    """
    if featmap.ndim == 4:
        f = sample2d
    elif featmap.ndim == 3:
        f = sample3d
    else:
        raise ValueError(f"Cannot handle featuremap dimensionality {featmap.ndim}.")

    return f(
        coords,
        featmap,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        encoder=encoder,
    )
