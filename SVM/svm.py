#!/usr/bin/env python3

from random import shuffle
from typing import Callable, Dict, Tuple

from dataset.continuous import AttributeName, Attributes, Example, Examples

Weights = Dict[str, float]

def predict(weights: Weights, bias: float, example: Example):
    return bias + sum(map(lambda k: weights[k] * example[k], weights.keys()))

def svm(
        T: int, C: float, gamma_schedule: Callable[[int], float],
        attributes: Attributes, label: AttributeName, examples: Examples
) -> Tuple[Weights, float]:
    N = len(examples)

    w = { k: 0.0 for k in attributes }
    b = 0.0

    def J(w: Weights, b: float, xi: Example):
        yi = 1 if xi[label] == 1 else -1
        # NOTE: Regularization term does not include bias in norm^2.
        regularization = 0.5 * sum(map(lambda n: n ** 2, w.values()))
        hinge_loss = max(0, 1 - yi * predict(w, b, xi))
        return regularization + C * N * hinge_loss

    for t in range(T):
        shuffle(examples)
        gamma = gamma_schedule(t)
        for xi in examples:
            prediction = predict(w, b, xi)
            yi = 1 if xi[label] == 1 else -1
            if yi * prediction <= 1:
                for k in w:
                    w[k] = w[k] - gamma * w[k] + gamma * C * N * yi * xi[k]
                b = b + gamma * C * N * yi * 1
            else:
                for k in w:
                    w[k] = (1 - gamma) * w[k]

    return w, b
