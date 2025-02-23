from .models import Base, Layer
from typing import List


class NeuralNetwork(Base):

    @property
    def layers(self):
        return self._layers

    def __init__(self):
        self._layers = None

    def __repr__(self):
        return (f'Network with {len(self._layers) - 2} hidden layers.'
                f' Input size: {self._layers[0].size}, output size:{self._layers[-1].size}')

    def setup(self, list_of_layers: List[Layer]):
        """Задает структуру слоев нейросети

        :return: None
        """
        self._layers = list_of_layers
        for previous_layer, layer in zip(self._layers[:-1], self._layers[1:]):
            layer.init_weights(previous_layer.size)

    def forward(self, x):
        if not self._layers:
            raise RuntimeError('The network has no layers')
        output = x.copy()
        for layer in self._layers:
            output = layer.forward(output)
        return output


    def backward(self, grad):
        pass

    def update(self, lr):
        pass

    def train(self, X, Y, epochs=1000, lr=0.1):
        """Обучает нейросеть с помощью градиентного спуска.

        Этот метод выполняет итеративное обучение нейронной сети, используя метод обратного распространения ошибки
        (backpropagation) и обновление параметров с помощью градиентного спуска.

        Алгоритм обучения:
            1. Для каждой эпохи:
                a) Прямое распространение (forward pass) для вычисления предсказания.
                b) Вычисление ошибки (функции потерь).
                c) Обратное распространение (backpropagation) для вычисления градиентов.
                d) Обновление параметров модели с использованием градиентного спуска.
            2. Повторять процесс до завершения указанного количества эпох.

        Args:
            X (numpy.ndarray): Входные данные размерности (n_samples, n_features), где:
                - n_samples — количество примеров в обучающем наборе.
                - n_features — количество признаков на один пример.
            Y (numpy.ndarray): Целевые значения размерности (n_samples, n_outputs), где:
                - n_outputs — количество выходных нейронов (размерность выходных данных).
            epochs (int, optional): Количество эпох обучения. Одна эпоха означает один полный проход по обучающим данным.
                По умолчанию равно 1000.
            lr (float, optional): Коэффициент скорости обучения (learning rate), определяющий шаг обновления параметров
                модели. По умолчанию равно 0.1.

        Returns:
            None

        Side Effects:
            - Изменяет веса и параметры модели в процессе обучения.
            - Может выводить информацию о процессе обучения (например, значение функции потерь).
        """
        pass