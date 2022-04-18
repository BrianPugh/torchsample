import torch
import torch.nn.functional as F


def sample2d(
    coords,
    featmap,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
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
        ``(b, c, h, w)``
    encoder : callable
        Converts coords to positional encoding.
        Concatenates the positional encoding to returned samples.
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
        output = output[:, :, 0]

    return output
