# %%

from gnp.nn import *
from IPython.display import clear_output

mse = MSE()

target = gnp.array(np.random.randn(2, 12))
optim = SGD(lr=0.00001)


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

# l = Dense(2, 1)

# l.weights["w"] = gnp.GNPArray([[1], [3]]) # (2, 1)
# l.weights["b"] = gnp.GNPArray([[6]]) # (1, 1)

l1 = Dense(1, 2)
l2 = Dense(2, 1)
model = Sequential([
  l1,
  # Tanh(),
  Relu(),
  l2
])

# a = gnp.GNPArray([[1], [2], [3]])
# # b = gnp.GNPArray([[0], [1]])
# model(a)



# %%

x_tensor = gnp.GNPArray(x[..., np.newaxis])
y_tensor = gnp.GNPArray(y[..., np.newaxis])

# %%

pred = model(x_tensor)
grad = MSE().grad(y_tensor, pred)
grad

# %%

grad = l2.backward(grad)
grad = l1.backward(grad)
grad

# %%

model.backward(grad)
optim.step(model)


model(x_tensor)

# %%

model.layers[2].weights

# %%



# %%

for i in range(100):
  pred = model(x_tensor)
  loss = MSE()(y_tensor, pred)
  grad = MSE().grad(y_tensor, pred)
  model.backward(grad)
  optim.step(model)

  # displaying
  clear_output(wait=True)
  plt.scatter(x, y)
  plt.plot(pred._data)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("Training Data")
  plt.show()

# %%

pred = model(x_tensor)


plt.scatter(x, y)
plt.plot(pred._data)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()

# %%


a = gnp.array(np.random.randn(1, 10))
b = gnp.array(np.random.randn(10, 6))
c = 2

a * gnp.GNPArray(32)



# %%

np.random.randn(10, 10).shape


# %%
from numba import cuda




