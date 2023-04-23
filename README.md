# GNP: Fast and Scalable Tensor Engine

* GNP (GPU Numpy) is a fast and scalable tensor computing library that utilize the parallel computation of GPUs. The library is itself built upon `numpy` library. `numpy` is a computing library that runs on CPU only, and `GNP` boost the performance of this framework by optimizing computation of numpy ndarray on GPU.

* GNP will have its own array structure which is based upon numpy. GNP implements GPU operations by binding python numpy data and use Cuda kernel to parallelize computation on GPU. The list of supported kernel is listed in the later section.

## Requirements and Installation:
* Requirements:
  * OS: Linux
  * Python version: 3.10
  * Anaconda

* Installation:
  ```
  # First, create a conda environment and install:
  conda create -n gnp pip python=3.10
  # install required libraries
  conda env update dev_env.yml
  # Instlal current library
  python setup.py install
  ```

## Project Structure and Supported APIs
### Supported Cuda Kernel
* Unary ops:
  * Negative
  * Positive
  * Invert (~x = 1/x)

* Binary ops:
  * add/subtract
  * multiply/true divide/floor divide
  * mod
  * pow
  * and
  * or
  * xor
  * right shift
  * left shift

* Comparison ops:
  * greater than
  * less than
  * greater than or equal to
  * less than or equal to
  * equal
  * not equal

* Linear algebra:
  * Matrix multiplication

### Supported APIs
* Experimental:
  * Neural network sequential API
  * Fully connected neural network forward and backward pass
  * SGD Optimizer, MSE Loss
  * Non-linear activations: Relu, Tanh, Sigmoid

## Result
* show code
* Compare time
* Future work

## Documentations

* The slide is here: [https://docs.google.com/presentation/d/1AYxhXKqORdddHHGxe6CHCLwE8b6S0u4CGl_jjvH4tIY/edit?usp=sharing](https://docs.google.com/presentation/d/1AYxhXKqORdddHHGxe6CHCLwE8b6S0u4CGl_jjvH4tIY/edit?usp=sharing)
  * The slide is also in the pdf file in the `docs` folder

* For the class diagrams, install `Visual Paradigm` and run the `GNP.vpp` file in the `docs/vpp/` folder
