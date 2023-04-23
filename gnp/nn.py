import gnp
import numpy as np
from gnp.type import ArrayLike
from typing import Callable, Iterator, Tuple, Sequence, Dict
from .type import ParamAndGradType

class Layer:
  def __init__(self) -> None:
    self.weights: Dict[str, gnp.GNPArray] = {}
    self.grads: Dict[str, gnp.GNPArray] = {}
    self.inputs: gnp.GNPArray = None

  def forward(self, inputs: gnp.GNPArray) -> gnp.GNPArray:
    raise NotImplementedError()

  def backward(self, grad: gnp.GNPArray) -> gnp.GNPArray:
    raise NotImplementedError()

  def __call__(self, inputs: ArrayLike) -> gnp.GNPArray:
    if isinstance(inputs, list):
      _inputs = gnp.array(inputs)
    elif isinstance(inputs, np.ndarray):
      _inputs = gnp.array(inputs)
    elif isinstance(inputs, gnp.GNPArray):
      _inputs = inputs
    else:
      raise NotImplementedError()
    return self.forward(_inputs)

class Dense(Layer):
  def __init__(self, input_size: int, output_size: int) -> None:
    super().__init__()
    self.weights["w"] = gnp.array(np.random.randn(input_size, output_size))
    self.weights["b"] = gnp.array(np.random.randn(1, output_size))

  def forward(self, inputs: gnp.GNPArray) -> gnp.GNPArray:
    self.inputs = inputs
    return inputs @ self.weights["w"] + self.weights["b"]
  
  def backward(self, grad: gnp.GNPArray) -> gnp.GNPArray:
    self.grads["w"] = self.inputs.T @ grad
    self.grads["b"] = np.sum(grad, axis=0)
    return grad @ self.weights["w"].T

class Activation(Layer):
  def __init__(
      self, 
      act_func: Callable[[gnp.GNPArray], gnp.GNPArray], \
      act_func_prime: Callable[[gnp.GNPArray], gnp.GNPArray]
  ) -> None:
    super().__init__()
    self.act_func = act_func
    self.act_func_prime = act_func_prime
  
  def forward(self, inputs: gnp.GNPArray) -> gnp.GNPArray:
    self.inputs = inputs
    return self.act_func(inputs)

  def backward(self, grad: gnp.GNPArray) -> gnp.GNPArray:
    return self.act_func_prime(self.inputs) * grad


def tanh(x: gnp.GNPArray) -> gnp.GNPArray:
  return np.tanh(x._data)

def tanh_prime(x: gnp.GNPArray) -> gnp.GNPArray:
  y = tanh(x._data)
  return 1 - y ** 2

class Tanh(Activation):
  def __init__(self):
    super().__init__(tanh, tanh_prime)

def sigmoid(x: gnp.GNPArray) -> gnp.GNPArray:
  return gnp.GNPArray(1.0 / (1 + np.exp(-x._data)))

def sigmoid_prime(x: gnp.GNPArray) -> gnp.GNPArray:
  sig_x = sigmoid(x)._data
  return gnp.GNPArray(sig_x * (1 - sig_x))

class Sigmoid(Activation):
  def __init__(self):
    super().__init__(sigmoid, sigmoid_prime)

class Loss:
  def loss(self, label: gnp.GNPArray, target: gnp.GNPArray) -> float:
    raise NotImplementedError()

  def grad(self, label: gnp.GNPArray, target: gnp.GNPArray) -> gnp.GNPArray:
    raise NotImplementedError()

class MSE(Loss):
  def __call__(self, label: gnp.GNPArray, target: gnp.GNPArray) -> float:
    return gnp.GNPArray(np.sum((label._data - target._data) ** 2))

  def grad(self, label: gnp.GNPArray, target: gnp.GNPArray) -> gnp.GNPArray:
    return gnp.GNPArray(2 * (label._data - target._data))
  
class Sequential:
  def __init__(self, layers: Sequence[Layer]) -> None:
    self.layers = layers

  def forward(self, inputs: gnp.GNPArray) -> gnp.GNPArray:
    _inputs = inputs.copy()
    for layer in self.layers:
      _inputs = layer(_inputs)
    return _inputs

  def backward(self, grad: gnp.GNPArray) -> gnp.GNPArray:
    _grad = grad.copy()
    for layer in reversed(self.layers):
      _grad = layer.backward(_grad)
    return _grad

  def __call__(self, inputs: ArrayLike):
    return self.forward(inputs)

  def params_and_grads(self) -> ParamAndGradType:
    for layer in self.layers:
      for name, param in layer.weights.items():
        grad = layer.grads[name]
        yield param, grad

class Optimizer:
  def step(self, neural_network: Sequential) -> None:
    raise NotImplementedError

class SGD(Optimizer):
  def __init__(self, lr: float = 0.01) -> None:
    self.lr = lr

  def step(self, neural_network: Sequential) -> None:
    for param, grad in neural_network.params_and_grads():
      param -=  grad * gnp.GNPArray(self.lr)

