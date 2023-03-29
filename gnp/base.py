# %%

import numpy as np
from gnp.type import ArrayLike


class GNPArray():
  def __init__(self, _array: ArrayLike) -> None:
    self._array: np.ndarray = np.asarray(_array)


def array(arr: ArrayLike):
  return GNPArray(arr)



