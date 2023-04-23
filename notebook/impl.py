# %%

import numpy as np
import gnp
import time

np.random.seed(10)

a = gnp.array(np.random.randn(500, 500, 500))
b = gnp.array(np.random.randn(500, 500, 500))

t0 = time.time()
c = a * b
t1 = time.time()
print(f"Time elapse: {t1 - t0}")


a = np.random.randn(500, 500, 500)
b = np.random.randn(500, 500, 500)
t0 = time.time()
c = a * b
t1 = time.time()
print(f"Time elapse: {t1 - t0}")

# %%


a = gnp.array(np.random.randn(1, 10))
b = gnp.array(np.random.randn(10, 6))

(a @ b).shape

# %%

