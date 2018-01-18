# TensorD: A Tensor Decomposition Library in Tensorflow

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









## Installation











##  Example







## Reference

```
[1] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat,G. Irving, M. Isard, 
    et al., Tensorflow:  A system for large-scale machine learning., in:  OSDI, Vol. 16, 2016, pp. 265-283.
[2] Using gpus, https://www.tensorflow.org/tutorials/using_gpu.
```





