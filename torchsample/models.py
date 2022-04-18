import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Multi Layer Perceptron.

    Applies the MLP to the final dimension of input tensor.
    """

    def __init__(self, *layers, activation=F.relu):
        """Construct an ``MLP`` with specified nodes-per-layer.

        Parameters
        ----------
        layers : tuple
            List of how many nodes each layer should have.
            Must be at least 2 long.
        activation : callable
            Activation function applied to all layers except
            the output layer. Defaults to ``relu``.
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
        # Remove the last activation
        layers = layers[:-1]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through linear and activation layers.

        Parameters
        ----------
        x : torch.Tensor
            (..., feat_in) shaped tensor.

        Returns
        -------
        torch.Tensor
            (..., feat_out) shaped tensor.
        """
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
