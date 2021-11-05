#!/usr/bin/env python3

from math import inf, sqrt
import random
from typing import Dict, List, Tuple

from dataset.continuous import AttributeName, Attributes, Example, Examples

Weights = Dict[AttributeName, float]
WeightsAndBias = Tuple[Weights, float]

def compute_error(w: Weights, bias: float, example: Example, label: AttributeName) -> float:
    return example[label] - (bias + sum(map(lambda k: w[k] * example[k], w.keys())))

def compute_loss_gradient(w: Weights, bias: float, examples: Examples, label: AttributeName) -> Tuple[float, Weights, float]:
    cost = 0.0
    gradient_w = { k: 0.0 for k in w.keys() }
    gradient_bias = 0.0
    for example in examples:
        error = compute_error(w, bias, example, label)
        cost += 0.5 * (error ** 2)
        for k in gradient_w.keys():
            gradient_w[k] -= error * example[k]
        gradient_bias -= error
    return cost, gradient_w, gradient_bias

def bgd(examples: Examples, attributes: Attributes, label: AttributeName, r: float, max_iterations: int) -> Tuple[List[float], Weights, float]:
    w = { a: 0.0 for a in attributes }
    bias = 0.0

    change = inf
    i = 0

    costs = []

    while i < 2 or change > (1 / (10 ** 6)):
        if i > max_iterations:
            raise Exception("diverges")
        i += 1

        cost, gradient_w, gradient_bias = compute_loss_gradient(w, bias, examples, label)
        costs.append(cost)

        sum_sqs = 0
        for k in w.keys():
            adjust = r * gradient_w[k]
            w[k] -= adjust
            sum_sqs += adjust ** 2
        change = sqrt(sum_sqs)

        bias -= r * gradient_bias

    return costs, w, bias

def sgd(examples: Examples, attributes: Attributes, label: AttributeName, r: float, max_iterations: int) -> Tuple[List[float], Weights, float]:
    w = { a: 0.0 for a in attributes }
    bias = 0.0

    i = 0
    costs = []
    change = inf

    while i < 2 or change > (1 / (10 ** 6)):
        if i > max_iterations:
#            raise Exception("diverges")
            break
        i += 1

        example = random.choice(examples)
        _, gradient_w, gradient_bias = compute_loss_gradient(w, bias, [example], label)

        sum_sqs = 0
        for k in w.keys():
            adjust = -r * gradient_w[k]
            w[k] += adjust
            sum_sqs += adjust ** 2
        bias -= r * gradient_bias

        cost, _, _ = compute_loss_gradient(w, bias, examples, label)
        costs.append(cost)
#        change = sqrt(sum_sqs)

        if cost == inf:
            raise Exception("diverges")

        if i % 10000 == 0:
            print(r, "iter", i, "cost", cost, change)

    return costs, w, bias
