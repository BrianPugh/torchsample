No Batch
========

Especially when preprocessing exemplars in a ``Dataset``, it may be preferable
to operate without a batch dimension (``Dataloader`` collates the data
downstream). TorchSample supports this usecase by either supplying ``0`` to
functions that expect a batch size, or by using the ``nobatch`` subfunction.

.. code-block:: python

   out = {}
   image = torch.rand(3, 480, 640)
   image = image[None]  # Add a singleton batch dimension
   out["coords"] = ts.coord.randint(1, 4096, (640, 480))  # (1, 4096, 2)
   out["rgb"] = ts.sample(out["coords"], image, mode="nearest")  # (1, 4096, 3)

   # Remove the singleton dimensions since Dataloader will do the batching
   out["coords"] = out["coords"][0]
   out["rgb"] = out["rgb"][0]

Not bad, but its more terse by using the ``nobatch`` feature:

.. code-block:: python

   out = {}
   image = torch.rand(3, 480, 640)
   out["coords"] = ts.coord.randint(0, 4096, (640, 480))  # (4096, 2)
   out["rgb"] = ts.sample.nobatch(out["coords"], image, mode="nearest")  # (4096, 3)
