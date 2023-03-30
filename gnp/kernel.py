
from numba import cuda

@cuda.jit
def kernel_invert(x, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = 1.0 / x[i]


@cuda.jit
def kernel_negate(x, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = -x[i]

@cuda.jit
def kernel_add(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] + y[i]

@cuda.jit
def kernel_subtract(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] - y[i]

@cuda.jit
def kernel_multiply(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] * y[i]

@cuda.jit
def kernel_divide(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] / y[i]

@cuda.jit
def kernel_floor_divide(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] // y[i]

@cuda.jit
def kernel_mod(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] % y[i]

@cuda.jit
def kernel_pow(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] ** y[i]

@cuda.jit
def kernel_rshift(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] >> y[i]

@cuda.jit
def kernel_lshift(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] << y[i]

@cuda.jit
def kernel_logical_and(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] and y[i]

@cuda.jit
def kernel_logical_or(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] or y[i]

@cuda.jit
def kernel_logical_xor(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = (x[i] and not y[i]) or (not x[i] and y[i])

@cuda.jit
def kernel_eq(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] == y[i]

@cuda.jit
def kernel_gt(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] > y[i]

@cuda.jit
def kernel_lt(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] < y[i]

@cuda.jit
def kernel_le(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] <= y[i]

@cuda.jit
def kernel_ge(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] >= y[i]

@cuda.jit
def kernel_ne(x, y, out):
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # this is the unique block ID within the 1D grid

  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid
  
  start = tx + ty * block_size
  stride = block_size * grid_size

  # assuming x and y inputs are same length
  for i in range(start, x.shape[0], stride):
    out[i] = x[i] != y[i]

@cuda.jit
def kernel_batch_matmul(x, y, out):
  batch_size = x.shape[0]
  n_rows = x.shape[1]
  n_cols = y.shape[2]

  row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
  col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

  n_x_col = x.shape[2]

  # if id < batch_size and row < n_rows and col < n_cols:
  batch_id_start = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
  stride = cuda.blockDim.z * cuda.gridDim.z

  if row < n_rows and col < n_cols:
    for batch_id in range(batch_id_start, batch_size, stride):
      # For each batch, compute a row and column dot product
      tmp = 0
      for i in range(n_x_col):
        tmp += x[batch_id, row, i] * y[batch_id, i, col]

      out[batch_id, row, col] = tmp
