Basic Operations
================

In this section, we will intrduce how to use TensorToolbox to perform basic operations on both matrics and tensors.

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



Kronecker Products
^^^^^^^^^^^^^^^^^^
DTensor:

tf.Tensor:



Khatri-Rao Products
^^^^^^^^^^^^^^^^^^^
DTensor:

tf.Tensor:









Basic Operations with Tensors
-----------------------------

Addition & Subtraction
^^^^^^^^^^^^^^^^^^^^^^
DTensor:

tf.Tensor:




Inner Products
^^^^^^^^^^^^^^
tf.Tensor:



Vectorization & Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tf.Tensor:



Vector to Tensor
^^^^^^^^^^^^^^^^
tf.Tensor:



Unfolding & Folding
^^^^^^^^^^^^^^^^^^^
*Unfolding*, also known as *matricization*, is the process of reordering the elements of an *N* -way array into a matrix.
Here, we call operation **mode-n matricize** as **unfold** in default.

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


