Basic Operations
================

Most operations we offer return results with :class:`tf.Tensor` form, except some build-in class methods in our module.

To begin with, load in operation module:

.. code-block:: python

   import factorizer.base.ops as ops


Basic Operations with Matrices
------------------------------


Hadamard Products
^^^^^^^^^^^^^^^^^
The *Hadamard product* is the elementwise matrix product. Given matrices :math:`\mathbf{A}` and :math:`\mathbf{B}`, both
of size :math:`\mathit{I} \times \mathit{J}`, their Hadamard product is denoted by :math:`\mathbf{A} \ast \mathbf{B}`.
The result is defined by

.. math::
   \mathbf{A} \ast \mathbf{B} =
   \left[
   \begin{matrix}
   a_{11}b_{11}                       & a_{12}b_{12}                      & \cdots  & a_{1 \mathit{J}}b_{1 \mathit{J}}\\
   a_{21}b_{21}                       & a_{22}b_{22}                      & \cdots  & a_{2 \mathit{J}}b_{2 \mathit{J}}\\
   \vdots                             & \vdots                            & \ddots  & \vdots\\
   a_{\mathit{I} 1}b_{\mathit{I} 1}   & a_{\mathit{I} 2}b_{\mathit{I} 2}  & \cdots  & a_{\mathit{I} \mathit{J}}b_{\mathit{I} \mathit{J}}
   \end{matrix}
   \right]

For instance, using matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` defined as

.. math::
   \mathbf{A} =
   \left[
   \begin{matrix}
   1  & 4  & 7\\
   2  & 5  & 8\\
   3  & 6  & 9
   \end{matrix}
   \right] , \quad \mathbf{B} = \left[
                         \begin{matrix}
   2 & 3 & 4\\
   2 & 3 & 4\\
   2 & 3 & 4
                         \end{matrix}
                         \right]

Using :class:`DTensor` to store matrices, :math:`\, \mathbf{A} \ast \mathbf{B}` can be performed as:

.. code-block:: python

   >>> A = DTensor(tf.constant([[1,4,7], [2,5,8],[3,6,9]]))
   >>> B = DTensor(tf.constant([[2,3,4], [2,3,4],[2,3,4]]))
   >>> result = A*B    # result is a DTensor with shape (3,3)
   >>> tf.Session().run(result.T)
   array([[ 2, 12, 28],
          [ 4, 15, 32],
          [ 6, 18, 36]], dtype=int32)

Using :class:`tf.Tensor` to store matrices, :math:`\, \mathbf{A} \ast \mathbf{B}` can be performed as:

.. code-block:: python

   >>> A = tf.constant([[1,4,7], [2,5,8],[3,6,9]])
   >>> B = tf.constant([[2,3,4], [2,3,4],[2,3,4]])
   >>> tf.Session().run(ops.hadamard([A,B]))
   array([[ 2, 12, 28],
          [ 4, 15, 32],
          [ 6, 18, 36]], dtype=int32)

:func:`hadamard` also supports the Hadamard products of more than two matrices:

.. code-block:: python

   >>> C = tf.constant(np.random.rand(3,3))
   >>> D = tf.constant(np.random.rand(3,3))
   >>> tf.Session().run(ops.hadamard([A, B, C, D], skip_matrices_index=[1]))
       # the result is equal to tf.Session().run(ops.hadamard([A, C, D]))

Kronecker Products
^^^^^^^^^^^^^^^^^^
The *Kronecker product* of matrices :math:`\, \mathbf{A} \in \mathbb{R}^{\mathit{I} \times \mathit{J}}`
and :math:`\mathbf{B} \in \mathbb{R}^{\mathit{K} \times \mathit{L}}` is denoted by :math:`\mathbf{A} \otimes \mathbf{B}`.
The result is a matrix of size :math:`(\mathit{IK}) \times (\mathit{JL})` (See Kolda's [1]_ for more details).

For example, matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is defined as

.. math::
   \mathbf{A} =
   \left[
   \begin{matrix}
   1   & 2   & 3   & 4\\
   5   & 6   & 7   & 8\\
   9   & 10  & 11  & 12
   \end{matrix}
   \right] , \quad \mathbf{B} = \left[
                                \begin{matrix}
   1 & 1 & 1 & 1 & 1\\
   2 & 2 & 2 & 2 & 2
                                \end{matrix}
                                \right]
To perform :math:`\mathbf{A} \otimes \mathbf{B}` with :class:`tf.Tensor` objects:

.. code-block:: python

   >>> A = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]])    # the shape of A is (3, 4)
   >>> B = tf.constant([[1,1,1,1,1],[2,2,2,2,2]])    # the shape of B is (2, 5)
   >>> tf.Session().run(ops.kron([A, B]))
   # the shape of result is (6, 20)
   array([[ 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4],
          [ 2,  2,  2,  2,  2,  4,  4,  4,  4,  4,  6,  6,  6,  6,  6,  8,  8,  8,  8,  8],
          [ 5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8],
          [10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16],
          [ 9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12],
          [18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 24, 24, 24, 24, 24]], dtype=int32)

To perform :math:`\mathbf{B} \otimes \mathbf{A}`:

.. code-block:: python

   >>> tf.Session().run(ops.kron([A, B], reverse=True))
   # the shape of result is (6, 20)
   array([[ 1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  5,  6,  7,  8,  5,  6,  7,  8,  5,  6,  7,  8,  5,  6,  7,  8],
          [ 9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12],
          [ 2,  4,  6,  8,  2,  4,  6,  8,  2,  4,  6,  8,  2,  4,  6,  8,  2,  4,  6,  8],
          [10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 16],
          [18, 20, 22, 24, 18, 20, 22, 24, 18, 20, 22, 24, 18, 20, 22, 24, 18, 20, 22, 24]], dtype=int32)

It might seem useless when using ``reverse=True`` to calculate the Kronecker product of two matrices, considering ``ops.kron([B, A])``
also do the same work, but it is considerable efficient to perform :math:`X_1 \otimes X_2 \otimes \cdots \otimes X_N` using ``reverse=True`` when
given a list of :class:`tf.Tensor` objects ``matrices = [X_1, X_2, ..., X_N]``:

.. code-block:: python

   >>> tf.Session().run(ops.kron(matrices, reverse=True))

If the matrices are given in :class:`DTensor` form:

.. code-block:: python

   >>> A = DTensor(tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))

Then :math:`\mathbf{A} \otimes \mathbf{B}` can be performed as:

.. code-block:: python

   >>> dtensor_B = DTensor(tf.constant([[1,1,1,1,1],[2,2,2,2,2]]))
   >>> tf.Session().run(A.kron(dtensor_B).T)    # A.kron(dtensor_B) returns a DTensor
   array([[ 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4],
          [ 2,  2,  2,  2,  2,  4,  4,  4,  4,  4,  6,  6,  6,  6,  6,  8,  8,  8,  8,  8],
          [ 5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8],
          [10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16],
          [ 9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12],
          [18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 24, 24, 24, 24, 24]], dtype=int32)

or

.. code-block:: python

   >>> tf_B = tf.constant([[1,1,1,1,1],[2,2,2,2,2]])
   >>> tf.Session().run(A.kron(tf_B).T)    # A.kron(tf_B) returns a DTensor
   array([[ 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4],
          [ 2,  2,  2,  2,  2,  4,  4,  4,  4,  4,  6,  6,  6,  6,  6,  8,  8,  8,  8,  8],
          [ 5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8],
          [10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16],
          [ 9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12],
          [18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 24, 24, 24, 24, 24]], dtype=int32)



Khatri-Rao Products
^^^^^^^^^^^^^^^^^^^
The *Khatri-Rao product* can be expressed in Kronecker product form. Given matrices :math:`\mathbf{A} \in \mathbb{R}^{\mathit{I} \times \mathit{K}}`
and :math:`\mathbf{B} \in \mathbb{R}^{\mathit{J} \times \mathit{K}}` , their Khatri-Rao product is denoted by :math:`\mathbf{A} \odot \mathbf{B}`.
The result is a matrix of size :math:`(\mathit{IJ}) \times (\mathit{K})` and defined by

.. math::
   \mathbf{A} \odot \mathbf{B} =
   \left[
   \begin{matrix}
   \mathbf{a}_1 \otimes \mathbf{b}_1 &  \mathbf{a}_2 \otimes \mathbf{b}_2  & \cdots  & \mathbf{a}_\mathit{K} \otimes \mathbf{b}_\mathit{K}
   \end{matrix}
   \right]

Let's take a look at matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` defined as

.. math::
   \mathbf{A} =
   \left[
   \begin{matrix}
   1   & 2   & 3   & 4\\
   5   & 6   & 7   & 8\\
   9   & 10  & 11  & 12
   \end{matrix}
   \right] , \quad \mathbf{B} = \left[
                                \begin{matrix}
   1 & 1 & 1 & 1\\
   2 & 2 & 2 & 2
                                \end{matrix}
                                \right]

To perform :math:`\mathbf{A} \odot \mathbf{B}`:

.. code-block:: python

   >>> A = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]])    # the shape of A is (3, 4)
   >>> B = tf.constant([[1,1,1,1],[2,2,2,2]])    # the shape of B is (2, 4)
   >>> tf.Session().run(ops.khatri([A, B]))
   # the shape of the result is (6, 4)
   array([[ 1,  2,  3,  4],
          [ 2,  4,  6,  8],
          [ 5,  6,  7,  8],
          [10, 12, 14, 16],
          [ 9, 10, 11, 12],
          [18, 20, 22, 24]], dtype=int32)

:func:`khatri` function also offers ``skip_matrices_index`` to ignore specific matrices in the computation. For example, given ``matrices = [A, B, C, D]`` to
calculate :math:`\mathbf{A} \odot \mathbf{B} \odot \mathbf{D}`:

.. code-block:: python

   >>> C = tf.constant(np.random.rand(4,4))
   >>> D = tf.constant(np.random.rand(5,4))
   >>> matrices = [A, B, C, D]
   >>> tf.Session().run(ops.khatri(matrices, skip_matrices_index=[2]))
   # the shape of the result is (30, 4)

To obtain the result of :math:`\mathbf{D} \odot \mathbf{C} \odot \mathbf{B} \odot \mathbf{A}`:

.. code-block:: python

   >>> tf.Session().run(ops.khatri(matrices, reverse=True))
   # the shape of the result is (120, 4)

:class:`DTensor` class also offers class method :func:`DTensor.khatri` which accepts only one single :class:`DTensor` object or :class:`tf.Tensor` object:

.. code-block:: python

   >>> A = DTensor(tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
   >>> B = tf.constant([[1,1,1,1],[2,2,2,2]])
   >>> tf.Session().run(A.khatri(B).T)
   # the shape of the result is (6, 4)
   array([[ 1,  2,  3,  4],
          [ 2,  4,  6,  8],
          [ 5,  6,  7,  8],
          [10, 12, 14, 16],
          [ 9, 10, 11, 12],
          [18, 20, 22, 24]], dtype=int32)




Basic Operations with Tensors
-----------------------------

Addition & Subtraction
^^^^^^^^^^^^^^^^^^^^^^


DTensor:

tf.Tensor:




Inner Products
^^^^^^^^^^^^^^
The *inner product* of two same-sized tensor :math:`\mathcal{X}, \mathcal{Y} \in \mathbb{R}^{\mathit{I}_1 \times \mathit{I}_2 \times \cdots \times \mathit{I}_N}`
is the sum of products of their entries, which can be denoted as :math:`\langle \mathcal{X} , \mathcal{Y} \rangle`.

Given tensor :math:`\mathcal{X}, \mathcal{Y} \in \mathbb{R}^\mathit{3 \times 3 \times 2}` defined by their
frontal slices:

.. math::
   X_1 =
   \left[
   \begin{matrix}
   1  & 4  & 7\\
   2  & 5  & 8\\
   3  & 6  & 9
   \end{matrix}
   \right] , \quad X_2 = \left[
                         \begin{matrix}
   10 & 13 & 16\\
   11 & 14 & 17\\
   12 & 15 & 18
                         \end{matrix}
                         \right]

.. math::
   Y_1 =
   \left[
   \begin{matrix}
   1  & 1  & 1\\
   1  & 1  & 1\\
   1  & 1  & 1
   \end{matrix}
   \right] , \quad Y_2 = \left[
                         \begin{matrix}
   1 & 1 & 1\\
   1 & 1 & 1\\
   1 & 1 & 1
                         \end{matrix}
                         \right]

.. code-block:: python

   >>> X = tf.constant(np.array([[[1,10],[4,13],[7,16]], [[2,11],[5,14],[8,17]], [[3,12],[6,15],[9,18]]]))    # the shape of X is (3, 3, 2)
   >>> Y = tf.constant(np.array([[[1,1],[1,1],[1,1]], [[1,1],[1,1],[1,1]], [[1,1],[1,1],[1,1]]]))    # the shape of Y is (3, 3, 2)

To calculate :math:`\langle \mathcal{X} , \mathcal{Y} \rangle`:

.. code-block:: python

   >>> tf.Session().run(ops.inner(X, Y))
   171



.. warning::
   Notice that :func:`ops.inner` function does not support implicit type-casting, so be careful when using tensors
   of different ``dtype`` !


Vectorization & Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tf.Tensor:



Vector to Tensor
^^^^^^^^^^^^^^^^
tf.Tensor:



Unfolding & Folding
^^^^^^^^^^^^^^^^^^^
*Unfolding*, also known as *matricization*, is the process of reordering the elements of an *N* -way array into a matrix.
Here we call operation **mode-n matricization** as **unfolding** in default.

DTensor:

tf.Tensor:




General Matricization
^^^^^^^^^^^^^^^^^^^^^
DTensor:

tf.Tensor:



The *n* -mode Products
^^^^^^^^^^^^^^^^^^^^^^
tf.Tensor:



Tensor Contraction
^^^^^^^^^^^^^^^^^^
DTensor:

tf.Tensor:



References
----------
.. [1] Tamara G. Kolda and Brett W. Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.




