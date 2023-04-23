# %%

from gnp.nn import *


mse = MSE()

target = gnp.array(np.random.randn(2, 12))
optim = SGD(lr=0.01)

# %%


import matplotlib.pyplot as plt

# Generating random linear data
# There will be 50 data points ranging from 0 to 50
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)
 
# Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)
 
n = len(x) # Number of data points

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()

# %%

model = Sequential([
  Dense(1, 10),
  Dense(10, 1)
])


# %%

model(x_tensor)

# %%

x_tensor = gnp.GNPArray(x[..., np.newaxis])
y_tensor = gnp.GNPArray(y[..., np.newaxis])


# %%

for i in range(10):
  pred = model(x_tensor)
  loss = MSE()(y_tensor, pred)
  grad = MSE().grad(y_tensor, pred)
  model.backward(grad)
  optim.step(model)

# %%

model(x_tensor)




# %%


# %%


a = gnp.array(np.random.randn(1, 10))
b = gnp.array(np.random.randn(10, 6))
c = 2

a * gnp.GNPArray(32)



# %%

np.random.randn(10, 10).shape


# %%
from numba import cuda




