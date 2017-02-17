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
Let's take a look at a 2-way tensor defined as below:

.. math::
   \mathcal{X} =
   \left[
   \begin{matrix}
   1  & 2  & 3  & 4\\
   5  & 6  & 7  & 8\\
   9  & 10 & 11 & 12
   \end{matrix}
   \right]

.. code-block:: python

   >>> X = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])    # create the 2-way tensor

This CP decomposition can factorize :math:`\mathcal{X}` into 2 component rank-one tensors, and the CP model can be expressed as

.. math::
   \mathcal{X} \approx
   [\![ \mathbf{A}, \mathbf{B} ]\!]
   \equiv
   \sum\limits_{r=1}^\mathit{R} \mathbf{a}_r \circ \mathbf{b}_r.

If we assume the columns of :math:`\mathbf{A}` and :math:`\mathbf{B}` are normalized to length one with the weights absorbed
into the vector :math:`\boldsymbol{\lambda}  \in \mathbb{R}^\mathit{R}` so that

.. math::
   \mathcal{X} \approx
   [\![ \boldsymbol{\lambda};\mathbf{A}, \mathbf{B} ]\!]
   \equiv
   \sum\limits_{r=1}^\mathit{R} \lambda_r \: \mathbf{a}_r \circ \mathbf{b}_r.

Here we use singular value decomposition (SVD) to obtain the factor matrices (CP decomposition actually can be considered
higher-order generation of matrix SVD) :

.. code-block:: python

   >>> u,s,v = np.linalg.svd(X, full_matrices=False)    # perform matrix SVD on tensor X



Refer to [1]_ for more mathematical details.

Tucker Tensor
-------------




References
----------
.. [1]





