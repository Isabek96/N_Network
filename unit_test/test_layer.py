from unittest import TestCase

import numpy as np

from n_network.models import Layer


class TestLayer(TestCase):

    def setUp(self):
        np.random.seed(seed=44)


    def test_init_layer(self):
        layer = Layer(m=8, n=4)
        self.assertEqual((8, 4), layer.weights.shape)
        self.assertEqual(8, layer.biases.size)

    def test_forward(self):
        x = np.random.randn(4)
        layer = Layer(m=8, n=4)
        output = layer.forward(x)
        self.assertEqual(8, output.size)
        self.assertNotEqual(0, output[0])
        self.assertEqual(0, output[1])
        self.assertEqual(0, output[2])
