from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .f_activation import relu


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


class Layer(Base):

    @property
    def biases(self):
        return self._biases

    @property
    def size(self):
        return self._biases.size

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        raise RuntimeError('The weights can not be set directly')

    def __repr__(self):
        return f'Layer ({self.size}) Input size: {self._weights.shape[1]}'

    def __init__(self, m: int = 1, activation: Callable = relu):
        """
        Конструктор класса, создает слой для полносвязной сети из нейронов (линейных)

        :param m: количество нейронов в слое
        :param activation: функция активации
        """
        # вектор весов будет задан позже, здесь резервируем переменную
        self._weights = None

        # инициализация смещений случайным образом
        self._biases = np.random.randn(m)

        # векторизации функции активации для дальнейшего применения
        self._activation = np.vectorize(activation)

    def init_weights(self, n: int = 1):
        """
        Инициализация весов

        :param n: размерность вектора входа
        :return: None
        """
        # инициализация весов случайный образом
        self._weights = np.random.randn(self.size, n)

    def forward(self, x):
        """
        Прямой проход слоя
        :param x: вектор входа
        :return: отклик слоя в виде вектора, которая содержит m координат
        """

        # расчет выходов нейронов в векторной форме
        layer_output = np.dot(self._weights, x) + self._biases

        # применения поэлементно функции активации к полученному вектору выходов
        return self._activation(layer_output)

    def backward(self, grad):
        pass

    def update(self, lr):
        pass


class InputLayer(Base):
    """
    Модель входного слоя, который имеет меньше свойств по сравнению с другими слоями.

    """
    @property
    def size(self):
        return self._size

    def __init__(self, m: int = 1):
        self._size = m

    def __repr__(self):
        return f'Input layer ({self.size})'

    def forward(self, x):
        return x

    def backward(self, grad):
        pass

    def update(self, lr):
        pass