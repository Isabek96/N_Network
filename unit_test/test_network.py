from unittest import TestCase

import numpy as np

from n_network.models import Layer, InputLayer
from n_network.network import NeuralNetwork


class TestNetwork(TestCase):

    def setUp(self):
        self._layers =  [InputLayer(m=4),
                         Layer(m=8),
                         Layer(m=4),
                         Layer(m=2)
                        ]
        self._x = np.array([1, 2, 2, 1])

    def test_init_network(self):
        network = NeuralNetwork()
        network.setup(list_of_layers=self._layers)

    def test_forward(self):
        network = NeuralNetwork()
        network.setup(list_of_layers=self._layers)
        one_forwarding_output = network.forward(self._x)
        pass
