Generating Coordinates
======================

To keep a simple, uniform, consistent API; everything in TorchSample
operates off of normalized coordinates in range ``[-1, 1]``, unless
explicitly stated otherwise.

Similarly, all coordinates are **always** in order ``(x, y, ...)``.
If the shape of a space is give, it's **always** in order ``(x, y, ...)``.
For example, the shape of a 4D tensor is typically expressed as
``(b, c, h, w)``. So, if a function needs the query space shape, the
argument would be ``(w, h)`` since ``w`` is associated with ``x`` and
``h`` is associated with ``y``.


Random Coordinates
------------------
If you just need to generate random coordinates, use ``ts.coord.rand``:

.. code-block:: python

   coords = ts.coord.rand(16, 4096, 2, device="cuda")
   assert coords.shape == (16, 4096, 2)

If you would like to generate coordinates that fall exactly on pixels for a
given resolution, you can use ``ts.coord.randint``. Despite the name, the
returned coordinates are still normalized and in range ``[-1. 1]``.

.. code-block:: python

   image = torch.rand(16, 3, 480, 640)
   coords = ts.coord.randint(16, 4096, (640, 480))

Similar to ``numpy``, we offer convenience functions ending in ``_like``
so that the caller doesn't have to juggle around shape.

.. code-block:: python

   image = torch.rand(16, 3, 480, 640)
   coords = ts.coord.randint_like(4096, image)  # (16, 4096, 2)


Comprehensive Coordinates
-------------------------
During inference time, its common to want to comprehensively query an
entire space. ``ts.coord.full`` operates on

TODO: make the ``full`` API consistent.

.. code-block:: python

   image = torch.random
   coords = ts.coord.full((640, 480))


Helpers
-------
To go back-and-forth between normalized and unnormalized coordinates, the
helper functions ``ts.coord.normalize`` and ``ts.coord.unnormalize`` are
available.

.. code-block:: python

   coords = ts.coord.rand(16, 4096, 2)  # range [-1, 1]
   # xrange [0, 639];  yrange [0, 479]
   unnormalized_coords = ts.coord.unnormalize(coords, (640, 480))
   renormalized_coords = ts.coord.normalize(coords, (640, 480))
   assert_close(coords, renormalized_coords)
