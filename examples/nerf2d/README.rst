NeRF 2D
=======
Training and querying an MLP to map coordinates to RGB values.


Results
^^^^^^^

Without positional encoding (200, 800, and 2000 iterations):

.. image:: output/polite_pos-enc=False/200.jpg
   :width: 30%

.. image:: output/polite_pos-enc=False/800.jpg
   :width: 30%

.. image:: output/polite_pos-enc=False/2000.jpg
   :width: 30%

With positional encoding (200, 800, and 2000 iterations):

.. image:: output/polite_pos-enc=True/200.jpg
   :width: 30%

.. image:: output/polite_pos-enc=True/800.jpg
   :width: 30%

.. image:: output/polite_pos-enc=True/2000.jpg
   :width: 30%


Usage
^^^^^

To run and recreate the images in the ``output/`` directory:

.. code-block:: bash

   python main.py
   python main.py --pos-enc

To see all available run options:

.. code-block:: bash

   $ python main.py --help
   usage: main.py [-h] [--input INPUT] [--batch-size BATCH_SIZE] [--lr LR]
                  [--iterations ITERATIONS] [--save-freq SAVE_FREQ] [--pos-enc]

   NeRF 2D Example.

   optional arguments:
     -h, --help            show this help message and exit
     --input INPUT         Input image to learn. (default: input/polite.jpg)
     --batch-size BATCH_SIZE
                           Number of samples per minibatch. (default: 16384)
     --lr LR               AdamW learning rate. (default: 0.0003)
     --iterations ITERATIONS
                           Number of training iterations. (default: 2000)
     --save-freq SAVE_FREQ
                           Every this many training iterations, perform a full query and save the
                           prediction. (default: 200)
     --pos-enc             Use gamma positional encoding. (default: False)
