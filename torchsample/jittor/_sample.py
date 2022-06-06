import jittor as jt

from .. import default
from ._nobatch import nobatch


def _unsqueeze_at(tensor, d, n):
    """Unsqueeze ``tensor`` ``n`` times at dimension ``d``."""
    return tensor[(slice(None),) * d + (None,) * n]


def _squeeze_at(tensor, d, n):
    """Squeeze ``tensor`` ``n`` times at dimension ``d``.

    Does NOT perform a check that the dimensions are singleton.
    """
    return tensor[(slice(None),) * d + (0,) * n]


@nobatch
def sample2d(
    coords,
    featmap,
    mode="bilinear",
    padding_mode=default.padding_mode,
    align_corners=default.align_corners,
    encoder=None,
    feat_last=True,
):
    """Sample a featmap at specified normalized coords.

    Parameters
    ----------
    coords : jittor.Var
        ``(b, samples, dim)`` or ``(b, h, w, dim)`` coordinates in range
        ``[-1, 1]`` where the last dimension represents ``(x, y)``
        or ``(x, y, z)``.
    featmap : jittor.Var
        ``(b, c, h, w)`` 2D featuremap.
    encoder : callable
        Converts coords to positional encoding.
        If provided, concatenates the positional encoding to the
        end of the returned samples.
    mode : str
        Interpolation method.
    padding_mode : str
        Defaults to ``"border"`` (different from jittor default ``"zeros"``).
    feat_last : bool
        Returned features are the last dimension.


    Returns
    -------
    jittor.Var
        ``(b, samples, c)`` or ``(b, h, w, c)``. Sampled featuremap.
        Features are last dimension.
    """
    if not (coords.ndim == 3 or coords.ndim == 4):
        raise ValueError(f"Unknown coords shape {coords.shape=}.")

    n_singleton = 4 - coords.ndim
    coords = _unsqueeze_at(coords, 1, n_singleton)  # (b, 1, n_samples, dim)

    # coords are 4D at this point.

    output = jt.nn.grid_sample(
        featmap,
        coords,
        padding_mode=padding_mode,
        mode=mode,
        align_corners=align_corners,
    )

    if encoder is not None:
        encoded = encoder(coords)
        encoded = encoded.permute(0, 3, 1, 2)
        output = jt.concat([output, encoded], 1)

    if feat_last:
        output = output.permute(0, 2, 3, 1)
        output = _squeeze_at(output, 1, n_singleton)
    else:
        output = _squeeze_at(output, 2, n_singleton)

    return output


@nobatch
def sample3d(
    coords,
    featmap,
    mode="bilinear",
    padding_mode="border",
    align_corners=default.align_corners,
    encoder=None,
    feat_last=True,
):
    """Sample a featmap at specified normalized coords.

    Parameters
    ----------
    coords : jittor.Var
        ``(b, samples, dim)`` or ``(b, h, w, dim)`` coordinates in range
        ``[-1, 1]`` where the last dimension represents ``(x, y)``
        or ``(x, y, z)``.
    featmap : jittor.Var
        ``(b, c, d, h, w)`` 3D featuremap.
    encoder : callable
        Converts coords to positional encoding.
        If provided, concatenates the positional encoding to the
        end of the returned samples.
    mode : str
        Interpolation method.
    padding_mode : str
        Defaults to ``"border"`` (different from jittor default ``"zeros"``).
    feat_last : bool
        Returned features are the last dimension.

    Returns
    -------
    jittor.Var
        ``(b, samples, c)`` or ``(b, d, h, w, c)``. Sampled featuremap.
        Features are last dimension.
    """
    if not (coords.ndim == 3 or coords.ndim == 4 or coords.ndim == 5):
        raise ValueError(f"Unknown coords shape {coords.shape=}.")

    n_singleton = 5 - coords.ndim
    coords = _unsqueeze_at(coords, 1, n_singleton)

    # coords are 5D at this point.

    output = jt.nn.grid_sample(
        featmap,
        coords,
        padding_mode=padding_mode,
        mode=mode,
        align_corners=align_corners,
    )

    if encoder is not None:
        encoded = encoder(coords)
        encoded = encoded.permute(0, 4, 1, 2, 3)
        output = jt.concat([output, encoded], 1)

    if feat_last:
        output = output.permute(0, 2, 3, 4, 1)
        output = _squeeze_at(output, 1, n_singleton)
    else:
        output = _squeeze_at(output, 2, n_singleton)

    return output


@nobatch
def sample(
    coords,
    featmap,
    mode="bilinear",
    padding_mode="border",
    align_corners=default.align_corners,
    encoder=None,
    feat_last=True,
):
    """Sample a featmap at specified normalized coords.

    Unified interface for both 2D and 3D operations.

    Parameters
    ----------
    coords : jittor.Var
        ``(b, samples, dim)`` or ``(b, h, w, dim)`` or ``(b, d, h, w, dim)``
        coordinates in range ``[-1, 1]`` where the last dimension represents
        ``(x, y)`` or ``(x, y, z)``.
    featmap : jittor.Var
        ``(b, c, h, w)`` (2D) or ``(b, c, d, h, w)`` (3D).
    encoder : callable
        Converts coords to positional encoding.
        If provided, concatenates the positional encoding to the
        end of the returned samples.
    mode : str
        Interpolation method.
    padding_mode : str
        Defaults to ``"border"`` (different from jittor default ``"zeros"``).
    feat_last : bool
        Returned features are the last dimension.

    Returns
    -------
    jittor.Var
        ``(b, samples, c)`` or ``(b, h, w, c)``. Sampled featuremap.
        Features are last dimension.
    """
    if featmap.ndim == 4:
        f = sample2d
    elif featmap.ndim == 5:
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
        feat_last=feat_last,
    )
