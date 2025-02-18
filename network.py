from models import Base, Layer
import numpy as np

class NeuralNetwork(Base): # Базовая модель для нейронных сетей
    def __init__(self):
        self.layers = [] # Список слоев нейросети


    def set_structure(self, layers):
        """Задает структуру слоев нейросети

        :return: None
        """
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x # Возвращает предсказание нейросети

    def backward(self, grad):
        for layer in reversed(self.layers): # Обратный проход
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers: # Обновление весов и смещений слоев с использованием гра��иентного спуска
            layer.update(lr)

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
        output = self.forward(X)

        # Вычисление ошибки (например, MSE)
        loss = np.mean((output - Y) ** 2)

        # Обратное распространение
        grad = 2 * (output - Y) / Y.shape[0]  # Градиент ошибки (для MSE)
        self.backward(grad)

        # Обновление весов
        self.update(lr)

        if epochs % 100 == 0:
            print(f"Epoch {epochs}, Loss: {loss:.4f}")