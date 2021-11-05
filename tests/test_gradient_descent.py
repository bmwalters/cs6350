#!/usr/bin/env python3

import unittest

from LinearRegression.gradient_descent import compute_loss_gradient

class TestGradientDescent(unittest.TestCase):
    def test_compute_loss_gradient(self):
        _, gradient_w, gradient_b = compute_loss_gradient(
            { "x1": -1, "x2": 1, "x3": -1 }, -1, [
                { "x1":  1, "x2": -1, "x3":  2, "y":  1 },
                { "x1":  1, "x2":  1, "x3":  3, "y":  4 },
                { "x1": -1, "x2":  1, "x3":  0, "y": -1 },
                { "x1":  1, "x2":  2, "x3": -4, "y": -2 },
                { "x1":  3, "x2": -1, "x3": -1, "y":  0 },
            ], "y"
        )
        self.assertAlmostEqual(gradient_w["x1"], -22, places=3)
        self.assertAlmostEqual(gradient_w["x2"], 16, places=3)
        self.assertAlmostEqual(gradient_w["x3"], -56, places=3)
        self.assertAlmostEqual(gradient_b, -10, places=3)

#        costs, w, b = bgd(
#            [
#                { "x1":  1, "x2": -1, "x3":  2, "y":  1 },
#                { "x1":  1, "x2":  1, "x3":  3, "y":  4 },
#                { "x1": -1, "x2":  1, "x3":  0, "y": -1 },
#                { "x1":  1, "x2":  2, "x3": -4, "y": -2 },
#                { "x1":  3, "x2": -1, "x3": -1, "y":  0 },
#            ], set(("x1", "x2", "x3")), "y", r=0.02, max_iterations=7
#        )
#        print(costs, w, b)
