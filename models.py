from abc import ABC, abstractmethod
import numpy as np
from f_activation import sigmoid_derivative, sigmoid

class Base(ABC): #Абстрактный базовый класс для всех компонентов нейронной сети.
    """Абстрактный базовый класс для компонентов нейронной сети.

    Этот класс определяет интерфейс, который должны реализовывать все компоненты
    нейросети, включая отдельные нейроны, слои и всю сеть в целом.

    Методы:
        forward(x): Выполняет прямое распространение (forward pass).
        backward(grad): Выполняет обратное распространение ошибки (backpropagation).
        update(lr): Обновляет параметры модели с использованием градиентов.
    """132

    @abstractmethod
    def forward(self, x):
        """Выполняет прямое распространение.

        Args:
            x (numpy.ndarray): Входные данные для слоя или нейрона.

        Returns:
            numpy.ndarray: Выходные данные после применения вычислений.
        """
        pass

    @abstractmethod
    def backward(self, grad):
        """Выполняет обратное распространение ошибки.

        Вычисляет градиенты параметров на основе градиента ошибки, переданного
        от следующего слоя.

        Args:
            grad (numpy.ndarray): Градиент ошибки, поступающий от следующего слоя.

        Returns:
            numpy.ndarray: Градиент, передаваемый дальше в предыдущий слой.
        """
        pass

    @abstractmethod
    def update(self, lr):
        """Обновляет параметры модели.

        Обновляет веса и другие обучаемые параметры с учетом вычисленных градиентов.

        Args:
            lr (float): Скорость обучения (learning rate), управляющая шагом обновления параметров.
        """
        pass



class Neuron(Base): # Нейрон

    def __init__(self, input_size: int):
        self.weights = np.random.randn(input_size) # Веса нейрона
        self.bias = np.random.randn() # Бias нейрона
        self.input = None # Входные данные нейрона
        self.output = None # Выходные данные нейрона


    def forward(self, *args, **kwargs):
        "Прямое распространение: взвешенная сумма + активация"
        self.input = args # Принимаем входные данные
        self.output = sigmoid(np.dot(self.input, self.weights) + self.bias) # Вычисляем выходные данные нейрона
        return self.output # Возвращаем выходные данные нейрона


    def backward(self, grad):
        dz = grad * sigmoid_derivative(self.output) # Вычисляем градиенты
        dw = np.dot(self.input.T, dz) # Вычисляем градиенты весов
        db = np.sum(dz) # Вычисляем градиенты биаса
        return dw, db # Возвращаем градиенты весов и биаса

    def update(self, *args, **kwargs):
        "Обновление весов"
        self.weights -= lr * self.weights # Обновляем веса
        self.bias -= lr * self.bias # Обновляем биас


class Layer(Base): # Нейронные слои

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def update(self, lr):
        pass