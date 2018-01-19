# *TensorD*: A Tensor Decomposition Library in Tensorflow

[![Build Status](https://travis-ci.org/Large-Scale-Tensor-Decomposition/tensorD.svg?branch=master)](https://travis-ci.org/Large-Scale-Tensor-Decomposition/tensorD)

Tensor:D

## What is *TensorD*?

*TensorD* is a Python tensor library built on Tensorflow  [1]. It provides basic decomposition methods, such as Tucker decomposition and CANDECOMP/PARAFAC (CP) decomposition, as well as new decomposition methods developed recently, for example, Pairwise Interaction Tensor Decomposition. 



*TensorD* is designed to be flexible, lightweight and scalable when used to transform idea into result as soon as possible in nowadays research. Based on Tensorflow, *TensorD* has several key features:

- **GPU Compatibility**: *TensorD* is completely built within TensorFlow, which enables all GPUs to be visible to the process [2] and flexible usage on GPU computation for acceleration.
- **Static Computation Graph**: *TensorD* runs in Static Computaiton Graph way, which means defining computation graph at first then running real computaion with dataflow. 
- **Light-weighted**: *TensorD* is written in Python which provides high-level implementations of mathematical operations. Acquiring small memory footprint, *TensorD* is friendly to install even on mobile devices.
- **High modularity of structure for extensibility**: *TensorD* has a modular structure which facilitates the expansion optionally. *TensorD* modulizes its code for the convenience of using its tensor classes, loss functions, basic operations and decomposition models separately as well as plugged together. 
- **High-level APIs**: The tensor decomposition part in *TensorD* is object-oriented and high-level interface on TensorFlow, which facilitates direct using. The purpose of such design is that users can make simple calls without knowing the details of implementations.
- **Open Source and MIT Licensed**: *TensorD* uses MIT license, and is an open source library in Tensorflow. Everyone can use and modify according to their own specific applications.





## Structure

![Structure of TensorD](https://github.com/Large-Scale-Tensor-Decomposition/tensorD/pictures/struct.png)

*TensorD*'s implementations of structure are clear and modular. The library structure is roughly contains three main modules: 

1) data processing module, providing interface to read and write sparse tensor in coordinate format, and a transformation between sparse and dense tensor.
2) the basic operation module, which assembled via the linear algebra in TensorFlow, providing basic matrix and tensor operations not only for tensor decomposition but also for other algorithms.
3) and the decomposition algorithm module, including common decomposition algorithms such as CP decomposition [3, 4, 5], Tucker decomposition [6, 7], NCP decomposition [8, 9] and NTucker decomposition [8, 10]





## Installation











##  Example







## Reference

[1] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat,G. Irving, M. Isard, et al., Tensorflow:  A system for large-scale machine learning., in:  OSDI, Vol. 16, 2016, pp. 265-283.

[2] Using gpus, https://www.tensorflow.org/tutorials/using_gpu .

[3] H. A. Kiers, Towards a standardized notation and terminologyin multiway analysis, Journal of chemometrics 14 (3) (2000)105–122.

[4] J. Mocks, Topographic components model for event-related potentials and some biophysical considerations, IEEE transactions on biomedical engineering 35 (6) (1988) 482–484.

[5] J. D. Carroll, J.-J. Chang, Analysis of individual differences inmultidimensional scaling via an n-way generalization of eckart-young decomposition, Psychometrika 35 (3) (1970) 283–319.

[6] F. L. Hitchcock, The expression of a tensor or a polyadic as asum of products, Studies in Applied Mathematics 6 (1-4) (1927)164–189.

[7] L. R. Tucker, Some mathematical notes on three-mode factoranalysis, Psychometrika 31 (3) (1966) 279–311.

[8] M. H. Van Benthem, M. R. Keenan, Fast algorithm for the solution of large-scale non-negativity-constrained least squares problems, Journal of chemometrics 18 (10) (2004) 441–450.

[9] P. Paatero, A weighted non-negative least squares algorithm for three-way parafacfactor analysis, Chemometrics and Intelligent Laboratory Systems 38 (2) (1997) 223–242.	


[10] Y.-D. Kim, S. Choi, Nonnegative tucker decomposition, in: Computer Vision and Pattern Recognition, 2007. CVPR’07. IEEE Conference on, IEEE, 2007, pp. 1–8.

