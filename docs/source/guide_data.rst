Data Types
==========

A *tensor* is a multidimensional array. For the sake of different applications, we will introduce 4 different data
types to store tensors.


Dense Tensor
------------
:class:`factorizer.base.DTensor` Class is used to store general high-order tensors, especially dense tensors.
This data type accepts 2 kinds of tensor data, both :class:`tf.Tensor` and :class:`np.ndarray`.

Let's take for this example the tensor :math:`\mathcal{X} \in \mathbb{R}^{3 \times 4 \times 2}` defined by its
frontal slices:

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

   >>> from factorizer.base.type import DTensor
   >>> tensor = np.array([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]])
   >>> dense_tensor = DTensor(tensor)

Creating :class:`DTensor` with :class:`tf.Tensor`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

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

   >>> X = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])    # the shape of tensor X is (3,4)

The CP decomposition can factorize :math:`\mathcal{X}` into 2 component rank-one tensors, and the CP model can be
expressed as

.. math::
   \mathcal{X} \approx
   [\![ \mathbf{A}, \mathbf{B} ]\!]
   \equiv
   \sum\limits_{r=1}^\mathit{R} \mathbf{a}_r \circ \mathbf{b}_r.

If we assume the columns of :math:`\mathbf{A}` and :math:`\mathbf{B}` are normalized to length one with the weights
absorbed into the vector :math:`\boldsymbol{\lambda}  \in \mathbb{R}^\mathit{R}` so that

.. math::
   \mathcal{X} \approx
   [\![ \boldsymbol{\lambda};\mathbf{A}, \mathbf{B} ]\!]
   \equiv
   \sum\limits_{r=1}^\mathit{R} \lambda_r \: \mathbf{a}_r \circ \mathbf{b}_r
where :math:`\mathbf{A} = [ \mathbf{a}_1, \cdots, \mathbf{a}_\mathit{R} ], \, \mathbf{B} = [ \mathbf{b}_1, \cdots, \mathbf{b}_\mathit{R} ]`.

Here we use singular value decomposition (SVD) to obtain the factor matrices (CP decomposition actually can be
considered higher-order generation of matrix SVD):

.. code-block:: python

   >>> from factorizer.base.type import KTensor
   >>> u,s,v = np.linalg.svd(X, full_matrices=False)    # X is equal to np.dot(u, np.dot(np.diag(s), v)), that is X = u * diag(s) * v

Then we use 2 factor matrices and :math:`\boldsymbol{\lambda}` to create a :class:`factorizer.base.KTensor` object:

.. code-block:: python

   >>> A = u    # the shape of A is (3,3)
   >>> B = v.T    # the shape of B is (4,3)
   >>> kruskal_tensor = KTensor([A, B], s)    # the shape of s is (3,)

Notice that the first argument ``factors`` is a list of :class:`tf.Tensor` objects or :class:`np.ndarray` objects
representing factor matrices, and the order of these matrices must be fixed.

If you want to get the factor matrices with :class:`factorizer.base.KTensor` object:

.. code-block:: python

   >>> kruskal_tensor.U
   [<tf.Tensor 'Const:0' shape=(3, 3) dtype=float64>,
    <tf.Tensor 'Const_1:0' shape=(4, 3) dtype=float64>]

If you want to get the vector :math:`\boldsymbol{\lambda}` with :class:`factorizer.base.KTensor` object:

.. code-block:: python

   >>> kruskal_tensor.lambdas
   <tf.Tensor 'Reshape:0' shape=(3, 1) dtype=float64>

If you want to get the original tensor with :class:`factorizer.base.KTensor` object:

.. code-block:: python

   >>> original_tensor = tf.Session().run(kruskal_tensor.extract())
   >>> original_tensor
   array([[  1.,   2.,   3.,   4.],
          [  5.,   6.,   7.,   8.],
          [  9.,  10.,  11.,  12.]])

To make sure ``original_tensor`` is equal to the tensor :math:`\mathcal{X}`, you just need to run:

.. code-block:: python

   >>> np.testing.assert_array_almost_equal(X, original_tensor)
   # no Traceback means these two np.ndarray objects are exactly the same

Refer to [1]_ for more mathematical details.

Tucker Tensor
-------------




References
----------
.. [1] Tamara G. Kolda and Brett W. Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.





