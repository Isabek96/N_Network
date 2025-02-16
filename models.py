from abc import ABC, abstractmethod
import numpy as np


class Base(ABC):
    """Абстрактный базовый класс для компонентов нейронной сети.

    Этот класс определяет интерфейс, который должны реализовывать все компоненты
    нейросети, включая отдельные нейроны, слои и всю сеть в целом.

    Методы:
        forward(x): Выполняет прямое распространение (forward pass).
        backward(grad): Выполняет обратное распространение ошибки (backpropagation).
        update(lr): Обновляет параметры модели с использованием градиентов.
    """

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



class Neuron(Base):

    def __init__(self, input_size: int):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def forward(self, *args, **kwargs):
        pass

    def backward(self, grad):
        pass

    def update(self, *args, **kwargs):
        pass


class Layer(Base):

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def update(self, lr):
        pass