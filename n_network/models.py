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
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        raise RuntimeError('The weights can not be set directly')

    def __init__(self, m: int = 1, n: int = 1, activation: Callable = relu):
        """
        Конструктор класса, создает слой для полносвязной сети из нейронов (линейных)
        :param m: количество нейронов в слое
        :param n: размерность вектора входа
        :param activation: функция активации
        """
        # инициализация весов случайный образом
        self._weights = np.random.randn(m, n)

        # инициализация смещений случайным образом
        self._biases = np.random.randn(m)

        # векторизации функции активации для дальнейшего применения
        self._activation = np.vectorize(activation)

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