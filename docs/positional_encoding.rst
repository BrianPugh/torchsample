.. _Positional Encoding:

Positional Encoding
===================

In `NeRF`_ they showed that neural networks perform poorly at high frequency detailed when directly operating
on coordinates. They were able to achieve significantly higher performance when encoding coordinates as a vector
of sinusoids. ``torchsample.encoding`` contains some common coordinate encoding methods.

All encoding functions take in a single required argument, ``coords`` and produce a new transformed coordinate
tensor.

.. code-block:: python

   coords = ts.coord.rand(16, 4096, 2)
   gamma_encoded_coords = ts.encoding.gamma(coords)  # (16, 4096, 40)


.. _NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis: https://arxiv.org/pdf/2003.08934.pdf
