|GHA tests| |Codecov report| |pre-commit| |black|

TorchSample
===========

Lightweight pytorch helpers for neural network sampling.

**WARNING: API is not yet stable. Use at your own risk!**

Introduction
------------
Sampling neural networks has become more and more common

* LIIF
* NeRF
* PointRend

PyTorch provides builtin functions that allow you to sample
coordinates, but they're not super intuitive to use.
Its very easy to get confusion over coordinates, indexing,
ordering, among other things.
``torchsample`` intends to make it dead simple so you can
focus on other parts of the model.

Design Decisions
----------------

* ``align_corners=True`` by default.
* Everything is in normalized coordinates ``[-1, 1]`` by default.
* Simple wrapper functions are provided (like ``ts.coord.rand``) are
  provided to make the intentions of calling code more clear.
* Try and mimic native ``pytorch`` and ``torchvision`` interfaces as
  much as possible.

TODO
----
* Most functionality was created with 2D use-cases in mind. Need to update
  APIs to take into account 3D use-cases.

Usage
-----

Training
^^^^^^^^
A common scenario is to randomly sample points from a featmap and
from the ground truth.

.. code-block:: python

  import torchsample as ts

  b, c, h, w = batch["image"].shape
  coords = ts.coord.rand(b, 4096, 2)  # (b, 4096, 2) where the last dim is (x, y)

  feat_map = encoder(batch["image"])  # (b, feat, h, w)
  sampled = ts.sample2d(coords, feat_map)  # (b, 4096, feat)
  gt_sample = ts.sample2d(coords, batch["gt"])

Inference
^^^^^^^^^
During inference, a comprehensive query of the network to form a complete
image is common.

.. code-block:: python

  import torch
  import torchsample as ts

  b, c, h, w = batch["image"].shape
  output = torch.zeros(b, 1, h, w)
  index_coords, norm_coords = ts.coord.full_like(h, w, 2)  # (1, h*w, 2)
  feat_map = encoder(batch["image"])  # (b, feat, h, w)
  output[index_coords] = model(feat_map[norm_coords])


Positional Encoding
^^^^^^^^^^^^^^^^^^^
Common positional encoding schemes are available.

.. code-block:: python

  import torchsample as ts

  coords = ts.coord.rand(b, 4096, 2)
  pos_enc = ts.encoding.gamma(coords)

A common task it concatenating the positional encoding to
sampled values. You can do this by passing a callable into
``ts.sample2d``:

.. code-block:: python

  import torchsample as ts

  encoder = ts.encoding.Gamma()
  sampled = ts.sample2d(coords, feat_map, encoder=encoder)


Models
^^^^^^
``torchsample`` has some common builtin models:

.. code-block:: python

  import torchsample as ts

  # Properly handles (..., feat) tensors.
  model = torch.models.MLP(256, 256, 512, 512, 1024, 1024, 1)


.. |GHA tests| image:: https://github.com/BrianPugh/torchsample/workflows/tests/badge.svg
   :target: https://github.com/BrianPugh/torchsample/actions?query=workflow%3Atests
   :alt: GHA Status
.. |Codecov report| image:: https://codecov.io/github/BrianPugh/torchsample/coverage.svg?branch=main
   :target: https://codecov.io/github/BrianPugh/torchsample?branch=main
   :alt: Coverage
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: black
