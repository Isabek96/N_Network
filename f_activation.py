import numpy as np

def sigmoid(x):
    # Сигмоидная функция активации
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Производная сигмоидной функции
    return x * (1 - x)

def relu(x):
    # Relu (Rectified Linear Unit) активация
    return np.maximum(0, x)

def relu_derivative(x):
    # Производная Relu активации
    return (x > 0).astype(float)

def tanh(x):
    # Tanh (Hyperbolic Tangent) активация
    return np.tanh(x)

def tanh_derivative(x):
    # Производная Tanh активации
    return 1 - np.square(x)
