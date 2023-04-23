# %%

import numpy as np
import gnp
import time
from numba import cuda
from gnp.kernel import kernel_batch_matmul

np.random.seed(10)

data_a = np.random.randn(600, 600, 600)
data_b = np.random.randn(600, 600, 600)

a = gnp.array(data_a)
b = gnp.array(data_b)

t0 = time.time()
_data = a._data.reshape((-1, *a._data.shape[-2:]))
d_data = cuda.to_device(_data)
_other = b._data.reshape((-1, *b._data.shape[-2:]))
d_other = cuda.to_device(_other)
out = np.empty(((*_data.shape[:-1], _other.shape[-1])))
d_out = cuda.to_device(out)
kernel_batch_matmul[(256, 256, 256), (8, 8, 8)](d_data, d_other, d_out)
cuda.synchronize()
gnp_result = gnp.array(d_out.copy_to_host().reshape((*a._data.shape[:-1], b._data.shape[-1])))
t1 = time.time()
print(f"GNP batch mat mul time elapse: {t1 - t0}")


a = data_a
b = data_b
t0 = time.time()
np_result = a @ b
t1 = time.time()
print(f"Numpy batch mat time elapse: {t1 - t0}")


correct = np.allclose(gnp_result._data, np_result)
print(f"The result of GNP is {'correct' if correct else 'incorrect'}")


