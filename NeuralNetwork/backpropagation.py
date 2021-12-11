#!/usr/bin/env python3

from math import e
from random import shuffle, gauss
from typing import Callable, List

from dataset.continuous import AttributeName, Examples

sigmoid = lambda x: 1.0 / (1.0 + e ** (-x))
dot = lambda a, b: sum(map(lambda ab: ab[0] * ab[1], zip(a, b)))

def backpropagation(
        w: List[List[List[float]]], x: List[float], ystar: float
):
    # forward pass
    z1 = [1] + list(map(lambda ws: sigmoid(dot(x, ws)), w[1][1:]))
    z2 = [1] + list(map(lambda ws: sigmoid(dot(z1, ws)), w[2][1:]))
    y = dot(z2, w[3][1])

    # back propagation
    dL = y - ystar
    dw3 = [[]] + [list(map(lambda zi: dL * zi, z2))]

    dz2 = list(map(lambda w31i: dL * w31i, w[3][1]))
    dw2 = [[]] + list(map(lambda i: list(map(lambda z1i: dz2[i] * z2[i] * (1 - z2[i]) * z1i, z1)), range(1, len(z2))))

    dz1 = list(map(lambda wi: sum(map(lambda zi: dz2[zi] * z2[zi] * (1 - z2[zi]) * w[2][zi][wi], range(1, len(z2)))), range(len(w[3][1]))))
    dw1 = [[]] + list(map(lambda i: list(map(lambda xi: dz1[i] * z1[i] * (1 - z1[i]) * xi, x)), range(1, len(z1))))

    return [[], dw1, dw2, dw3]

def predict(w: List[List[List[float]]], x: List[float]):
    # forward pass
    z1 = [1] + list(map(lambda ws: sigmoid(dot(x, ws)), w[1][1:]))
    z2 = [1] + list(map(lambda ws: sigmoid(dot(z1, ws)), w[2][1:]))
    y = dot(z2, w[3][1])
    return y

def cost(examples: Examples, attributes: List[AttributeName], label: AttributeName, w: List[List[List[float]]]) -> float:
    avg_cost = 0.0
    for example in examples:
        x = [1.0] + list(map(lambda a: example[a], attributes))
        ystar = -1 if example[label] == 0 else 1
        y = predict(w, x)
        L = ((y - ystar) ** 2)/2
        avg_cost += L / len(examples)
    return avg_cost

def train_sgd(
        examples: Examples, attributes: List[AttributeName], label: AttributeName,
        width: int, T: int, r0: float, d: float, initial_weight: Callable[[], float]
):
    w = [
        [],
        [[]] + [[initial_weight() for _ in attributes] for _ in range(width - 1)],
        [[]] + [[initial_weight() for _ in range(width)] for _ in range(width - 1)],
        [[]] + [[initial_weight() for _ in range(width)]]
    ]

    for t in range(T):
        shuffle(examples)

        r = r0 / (1 + (r0/d) * t)

        for example in examples:
            x = [1.0] + list(map(lambda a: example[a], attributes))
            y = -1 if example[label] == 0 else 1

            gradient_w = backpropagation(w, x, y)

            # sum_sq_updates = 0

            for layer, layer_grad in zip(w, gradient_w):
                for ws, ws_grad in zip(layer, layer_grad):
                    for i in range(len(ws)):
                        ws[i] -= r * ws_grad[i]
                        # sum_sq_updates += (r * ws_grad[i]) ** 2

            # change = sqrt(sum_sq_updates)

        if t % (T / 10) == 0:
            print("epoch", t, "cost", cost(examples, attributes, label, w))

    return w
