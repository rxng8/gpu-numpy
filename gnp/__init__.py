"""
@Creator: Viet Dung Nguyen
@Date: March 28, 2023
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Setup for tools
"""

# from .base import array
# from .type import ArrayLike

# from gnpc import *
from .base import array, GNPArray
from .nn import (
  Layer,
  Dense,
  Activation,
  Tanh,
  Sigmoid,
  Relu,
  Loss,
  MSE,
  Sequential,
  Optimizer,
  SGD
)