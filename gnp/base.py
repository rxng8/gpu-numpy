"""
@Creator: Viet Dung Nguyen
@Date: March 28, 2023
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Setup for tools
"""


import numpy as np
# import gnp
from .type import ArrayLike


class GNPArray:
  def __init__(self, _array: ArrayLike) -> None:
    self._array: np.ndarray = np.asarray(_array)


def array(arr: ArrayLike):
  return GNPArray(arr)



