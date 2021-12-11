#!/usr/bin/env python3

import unittest

from NeuralNetwork.backpropagation import backpropagation

class TestGradientDescent(unittest.TestCase):
    def test_compute_loss_gradient(self):
        w = [
            [],
            [[], [-1, -2, -3], [1, 2, 3]],
            [[], [-1, -2, -3], [1, 2, 3]],
            [[], [-1, 2, -1.5]]
        ]
        x = [1.0, 1.0, 1.0]
        y = 1.0
        gradient_w = backpropagation(w, x, y)

        self.assertAlmostEqual(gradient_w[1][1][0], 0.001051, places=6)
        self.assertAlmostEqual(gradient_w[1][1][1], 0.001051, places=6)
        self.assertAlmostEqual(gradient_w[1][1][2], 0.001051, places=6)

        self.assertAlmostEqual(gradient_w[1][2][0], 0.001576, places=6)
        self.assertAlmostEqual(gradient_w[1][2][1], 0.001576, places=6)
        self.assertAlmostEqual(gradient_w[1][2][2], 0.001576, places=6)

        self.assertAlmostEqual(gradient_w[2][1][0], -0.1217, places=4)
        self.assertAlmostEqual(gradient_w[2][1][1], -0.0003009, places=7)
        self.assertAlmostEqual(gradient_w[2][1][2], -0.1214, places=4)

        self.assertAlmostEqual(gradient_w[2][2][0], 0.09127, places=5)
        self.assertAlmostEqual(gradient_w[2][2][1], 0.0002257, places=7)
        self.assertAlmostEqual(gradient_w[2][2][2], 0.09105, places=5)

        self.assertAlmostEqual(gradient_w[3][1][0], -3.437, places=3)
        self.assertAlmostEqual(gradient_w[3][1][1], -0.06197, places=5)
        self.assertAlmostEqual(gradient_w[3][1][2], -3.375, places=3)
