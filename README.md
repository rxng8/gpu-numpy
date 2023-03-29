# GNP: Fast and Scalable Tensor Engine

* GNP is a fast and scalable tensor computing library that utilize the parallel computation of GPUs. The library is itself built upon `numpy` library. `numpy` is a computing library that runs on CPU only, and `GNP` boost the performance of this framework by optimizing computation of numpy ndarray on GPU.

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
  pip install -e .
  ```

## Project Structure and Supported APIs
* Supported APIs
  * 