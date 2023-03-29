"""
@Creator: Viet Dung Nguyen
@Date: March 28, 2023
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Setup for tools
"""


import numpy as np
from .type import ArrayLike

class GNPArray:
  def __init__(self, _array: ArrayLike) -> None:
    self._array: np.ndarray = np.asarray(_array)
  
  #### Unary operators
  def __neg__(self):
    pass

  def __pos__(self):
    pass

  def __invert__(self):
    pass

  #### Comparison operators
  def __lt__(self, other):
    pass

  def __gt__(self, other):
    pass

  def __le__(self, other):
    pass

  def __ge__(self, other):
    pass

  def __eq__(self, other):
    pass

  def __ne__(self, other):
    pass

  #### Binary operators
  def __add__(self, other):
    pass

  def __sub__(self, other):
    pass

  def __mul__(self, other):
    pass

  def __truediv__(self, other):
    pass

  def __floordiv__(self, other):
    pass

  def __mod__(self, other):
    pass

  def __pow__(self, other):
    pass

  def __rshift__(self, other):
    pass
  
  def __lshift__(self, other):
    pass

  def __and__(self, other):
    pass

  def __or__(self, other):
    pass

  def __xor__(self, other):
    pass

  #### Asisgnment operators
  def __isub__(self, other):
    pass

  def __iadd__(self, other):
    pass

  def __imul__(self, other):
    pass

  def __idiv__(self, other):
    pass


  def __ifloordiv__(self, other):
    pass


  def __imod__(self, other):
    pass

  def __ipow__(self, other):
    pass

  def __irshift__(self, other):
    pass
  
  def __ilshift__(self, other):
    pass

  def __iand__(self, other):
    pass

  def __ior__(self, other):
    pass

  def __ixor__(self, other):
    pass

  #### Matmul
  def __matmul__(self, other):
    pass

  


def array(arr: ArrayLike):
  return GNPArray(arr)



