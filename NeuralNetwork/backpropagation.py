#!/usr/bin/env python3

from math import e
from typing import List

sigmoid = lambda x: 1.0 / (1.0 + e ** (-x))
dot = lambda a, b: sum(map(lambda ab: ab[0] * ab[1], zip(a, b)))

def backpropagation(
        w: List[List[List[float]]], x: List[float], ystar: float
):
    # forward pass
    z1 = [1] + list(map(lambda ws: sigmoid(dot(x, ws)), w[1][1:]))
    z2 = [1] + list(map(lambda ws: sigmoid(dot(z1, ws)), w[2][1:]))
    y = dot(z2, w[3][1])

    L = ((y - ystar) ** 2)/2

    dL = y - ystar
    dw3 = [[]] + [list(map(lambda zi: dL * zi, z2))]

    dz2 = list(map(lambda w31i: dL * w31i, w[3][1]))
    dw2 = [[]] + list(map(lambda i: list(map(lambda z1i: dz2[i] * z2[i] * (1 - z2[i]) * z1i, z1)), range(1, len(z2))))

    dz1 = list(map(lambda wi: sum(map(lambda zi: dz2[zi] * z2[zi] * (1 - z2[zi]) * w[2][zi][wi], range(1, len(z2)))), range(len(w[3][1]))))
    dw1 = [[]] + list(map(lambda i: list(map(lambda xi: dz1[i] * z1[i] * (1 - z1[i]) * xi, x)), range(1, len(z1))))

    return [[], dw1, dw2, dw3]
