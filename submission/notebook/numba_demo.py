# %%

import numpy as np
import numba
from numba import cuda

@cuda.jit
def increment_by_one(an_array):
  # Thread id in a 1D block
  tx = cuda.threadIdx.x
  # Block id in a 1D grid
  ty = cuda.blockIdx.x
  # Block width, i.e. number of threads per block
  bw = cuda.blockDim.x
  # Compute flattened index inside the array
  pos = tx + ty * bw
  if pos < an_array.size:  # Check array boundaries
      an_array[pos] += 1

an_array = np.asarray([1,2,3])

threadsperblock = 32
blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
a = increment_by_one[blockspergrid, threadsperblock](an_array)
an_array

# (2, 3, 4)
