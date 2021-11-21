#!/usr/bin/env python3

import random
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
        random.shuffle(examples)
        gamma = gamma_schedule(t)
        for xi in examples:
            prediction = predict(w, b, xi)
            yi = 1 if xi[label] == 1 else -1

            print("=== sub gradients ===")
            if yi * prediction <= 1:
                for k in w:
                    print("sub-gradient", k, w[k] - C * N * yi * xi[k])
                    w[k] = w[k] - gamma * w[k] + gamma * C * N * yi * xi[k]
                print("sub-gradient\tbias", 0 - C * N * yi * 1)
                b = b + gamma * C * N * yi * 1
            else:
                for k in w:
                    print("sub-gradient", k, w[k])
                    w[k] = (1 - gamma) * w[k]
                print("sub-gradient\tbias", 0)

    return w, b

def main():
    random.seed(4242)

    attributes = ["x1", "x2", "x3"]
    label = "y"
    examples = [
        { "x1": 0.5, "x2": -1, "x3": 0.3, "y": 1 },
        { "x1": -1, "x2": -2, "x3": -2, "y": -1 },
        { "x1": 1.5, "x2": 0.2, "x3": -2.5, "y": 1 },
    ]

    def gamma_schedule(t: int):
        return [0.01, 0.005, 0.0025][t]

    C = 1/3
    weights, bias = svm(3, C, gamma_schedule, set(attributes), label, examples)

if __name__ == "__main__": pass
if __name__ == "__main__": main()
