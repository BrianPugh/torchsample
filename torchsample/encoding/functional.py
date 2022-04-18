import torch


def identity(coords):
    """ Returns ``coords`` as-is.
    """
    return coords


def gamma(coords, order=10, cat=True):
    """ Positional encoding via sin and cos.

    From:
        NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

    Parameters
    ----------
    coords : torch.Tensor
        (b, samples, dim) Coordinates to convert to positional encoding.
        In range ``[-1, 1]``.
    order : int
        Number
    cat : bool
        If ``true``, return a single torch.Tensor.
        Otherwise, returns a list of tensors.
        Saves a tiny bit of memory if you are going to concatenate with
        other features anyways.

    Returns
    -------
    torch.Tensor
        (b, samples, 2*dim*order)
    """
    output = []
    for o in range(order):
        freq = (2**o) * torch.pi
        cos = torch.cos(freq * coords)
        sin = torch.sin(freq * coords)
        output.append(cos)
        output.append(sin)
    if cat:
        output = torch.cat(output, dim=-1)
    return output

def nearest_pixel(coords):
    """Returns offset to nearest pixel.

    From:
        Learning Continuous Image Representation with Local Implicit Image Function
    """
