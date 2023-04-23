"""
@Creator: Viet Dung Nguyen
@Date: March 28, 2023
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Setup for tools
"""


import numpy as np
from numba import cuda

from .type import ArrayLike
from .kernel import (
  kernel_batch_matmul,
  kernel_divide,
  kernel_eq,
  kernel_floor_divide,
  kernel_ge,
  kernel_gt,
  kernel_invert,
  kernel_logical_and,
  kernel_logical_or,
  kernel_logical_xor,
  kernel_lshift,
  kernel_lt,
  kernel_mod,
  kernel_ne,
  kernel_negate,
  kernel_add,
  kernel_pow,
  kernel_rshift,
  kernel_subtract,
  kernel_multiply,
  kernel_divide_const,
  kernel_eq_const,
  kernel_ne_const,
  kernel_floor_divide_const,
  kernel_ge_const,
  kernel_gt_const,
  kernel_logical_and_const,
  kernel_logical_or_const,
  kernel_logical_xor_const,
  kernel_lshift_const,
  kernel_lt_const,
  kernel_mod_const,
  kernel_add_const,
  kernel_pow_const,
  kernel_rshift_const,
  kernel_subtract_const,
  kernel_multiply_const
)


def array(arr: ArrayLike):
  return GNPArray(arr)



class GNPArray:

  # Static variable
  threads_per_block = 512
  blocks_per_grid = 131072
  
  def __init__(self, data: ArrayLike) -> None:
    self._data: np.ndarray = None
    if isinstance(data, GNPArray):
      self._data = data._data
    else:
      self._data = np.asarray(data)

  def __str__(self):
    return f"GNPArray({self._data})"

  def copy(self):
    return GNPArray(self._data)

  @property
  def shape(self):
    return self._data.shape
  
  @property
  def T(self):
    return GNPArray(self._data.T)
    
  #### Unary operators
  def __neg__(self):
    d_data = cuda.to_device(np.ravel(self._data))
    d_out = cuda.device_array_like(d_data)
    kernel_negate[self.blocks_per_grid, self.threads_per_block](d_data, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __pos__(self):
    return GNPArray(self._data)

  def __invert__(self):
    d_data = cuda.to_device(np.ravel(self._data))
    d_out = cuda.device_array_like(d_data)
    kernel_invert[self.blocks_per_grid, self.threads_per_block](d_data, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  #### Comparison operators
  def __lt__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_lt_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_lt[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __gt__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_gt_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_gt[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __le__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_lshift_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_lshift[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __ge__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_ge_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_ge[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __eq__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_eq_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_eq[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __ne__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_ne_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_ne[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  #### Binary operators
  def __add__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_add_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_add[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __sub__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_subtract_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_subtract[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __mul__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_multiply_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_multiply[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __truediv__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_divide_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_divide[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __floordiv__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_floor_divide_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_floor_divide[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __mod__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_mod_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_mod[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __pow__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_pow_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_pow[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __rshift__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_rshift_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_rshift[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))
  
  def __lshift__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_lshift_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_lshift[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __and__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_logical_and_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_logical_and[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __or__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_logical_or_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_logical_or[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  def __xor__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    d_out = cuda.device_array_like(d_data)
    if len(other.shape) == 1:
      kernel_logical_xor_const[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    else:
      kernel_logical_xor[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_out)
    cuda.synchronize()
    return GNPArray(d_out.copy_to_host().reshape(self._data.shape))

  #### Asisgnment operators
  def __isub__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_subtract[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __iadd__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_add[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __imul__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_multiply[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __idiv__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_divide[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self


  def __ifloordiv__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_floor_divide[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __imod__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_mod[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __ipow__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_pow[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __irshift__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_rshift[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self
  
  def __ilshift__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_lshift[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __iand__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_logical_and[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __ior__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_logical_or[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  def __ixor__(self, other):
    d_data = cuda.to_device(np.ravel(self._data))
    d_other = cuda.to_device(np.ravel(other._data))
    kernel_logical_xor[self.blocks_per_grid, self.threads_per_block](d_data, d_other, d_data)
    cuda.synchronize()
    self._data = d_data.copy_to_host().reshape(self._data.shape)
    return self

  #### Matmul
  def __matmul__(self, other):
    if len(self._data.shape) == 2:
      _ndata = self._data[np.newaxis, ...]
      _data = _ndata.reshape((-1, *_ndata.shape[-2:]))
      d_data = cuda.to_device(_data)
      _nother = other._data[np.newaxis]
      _other = _nother.reshape((-1, *_nother.shape[-2:]))
      d_other = cuda.to_device(_other)
      out = np.empty(((*_data.shape[:-1], _nother.shape[-1])))
      d_out = cuda.to_device(out)
      kernel_batch_matmul[(256, 256, 256), (8, 8, 8)](d_data, d_other, d_out)
      cuda.synchronize()
      return GNPArray(d_out.copy_to_host().reshape((*_ndata.shape[:-1], _nother.shape[-1])).squeeze(0))
    else:
      _data = self._data.reshape((-1, *self._data.shape[-2:]))
      d_data = cuda.to_device(_data)
      _other = other._data.reshape((-1, *other._data.shape[-2:]))
      d_other = cuda.to_device(_other)
      out = np.empty(((*_data.shape[:-1], _other.shape[-1])))
      d_out = cuda.to_device(out)
      kernel_batch_matmul[(256, 256, 256), (8, 8, 8)](d_data, d_other, d_out)
      cuda.synchronize()
      return GNPArray(d_out.copy_to_host().reshape((*self._data.shape[:-1], other._data.shape[-1])))

  def __imatmul__(self, other):
    if len(self._data.shape) == 2:
      _ndata = self._data[np.newaxis, ...]
      _data = _ndata.reshape((-1, *_ndata.shape[-2:]))
      d_data = cuda.to_device(_data)
      _nother = other._data[np.newaxis]
      _other = _nother.reshape((-1, *_nother.shape[-2:]))
      d_other = cuda.to_device(_other)
      out = np.empty(((*_data.shape[:-1], _nother.shape[-1])))
      d_out = cuda.to_device(out)
      kernel_batch_matmul[(256, 256, 256), (8, 8, 8)](d_data, d_other, d_out)
      cuda.synchronize()
      self._data = d_out.copy_to_host().reshape((*_ndata.shape[:-1], _nother.shape[-1])).squeeze(0)
    else:
      _data = self._data.reshape((-1, *self._data.shape[-2:]))
      d_data = cuda.to_device(_data)
      _other = other._data.reshape((-1, *other._data.shape[-2:]))
      d_other = cuda.to_device(_other)
      out = np.empty(((*_data.shape[:-1], _other.shape[-1])))
      d_out = cuda.to_device(out)
      kernel_batch_matmul[(256, 256, 256), (8, 8, 8)](d_data, d_other, d_out)
      cuda.synchronize()
      self._data = d_out.copy_to_host().reshape((*self._data.shape[:-1], other._data.shape[-1]))
    return self
  
  def __repr__(self) -> str:
    return f"{self._data}"