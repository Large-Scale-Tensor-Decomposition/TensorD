Basic Operations
================

In this section, we will intrduce how to use TensorToolbox to perform basic operations on both matrics and tensors.


.. code-block:: python

   DTensor: unfold(张量->矩阵), t2mat(张量->矩阵),
            kron(张量乘法), khatri(张量间乘法),
            加法, Hadamard乘法(同形张量),
            减法, fold(矩阵->张量)

   basic: unfold(张量->矩阵), fold(矩阵->张量),
          t2mat(矩阵->张量「按任意指定轴顺序」),
          vectorize(张量->向量), vec_to_tensor,
          mul(张量间乘法tensor contraction),
          inner(同形张量间内积), ttm(张量x一堆矩阵),
          hadamard(一堆同形矩阵对应元素相乘),
          kron(矩阵), khatri(矩阵)


Basic Operations with Matrices
------------------------------




Basic Operations with Tensors
-----------------------------


