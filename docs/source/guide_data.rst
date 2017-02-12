Data Types
==========

A *tensor* is a multidimensional array. For the sake of different applications, we will introduce 4 different data types to store tensors.


Dense Tensor
------------
:class:`factorizer.base.DTensor` Class is used to store general high-order tensors, especially dense tensors. This data type accepts 2 kinds of tensor data, both :class:`tf.Tensor` and :class:`np.ndarray`.

Let's take for this example the tensor :math:`\mathcal{X} \in \mathbb{R}^{3 \times 4 \times 2}` defined by its frontal slices:

.. math::
   X_1 =
   \left[
   \begin{matrix}
   1  & 4  & 7  & 10\\
   2  & 5  & 8  & 11\\
   3  & 6  & 9  & 12
   \end{matrix}
   \right] , \quad X_2 = \left[
                         \begin{matrix}
   13 & 16 & 19 & 22\\
   14 & 17 & 20 & 23\\
   15 & 18 & 21 & 24
                         \end{matrix}
                         \right]

Creating :class:`DTensor` with :class:`np.ndarray`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   >>> import numpy as np
   >>> from factorizer.base.type import DTensor
   >>> tensor = np.array([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]])
   >>> dense_tensor = DTensor(tensor)

Creating :class:`DTensor` with :class:`tf.Tensor`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   >>> import tensorflow as tf
   >>> from factorizer.base.type import DTensor
   >>> tensor = tf.constant([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]])
   >>> dense_tensor = DTensor(tensor)

Kruskal Tensor
--------------
:class:`factorizer.base.KTensor` Class is designed for Kruskal tensors.

Refer to [1]_ for more mathematical details.


Tucker Tensor
-------------




References
----------
.. [1]





