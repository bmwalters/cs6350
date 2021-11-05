#!/usr/bin/env python3

from random import shuffle
from typing import Dict, Iterable, List, Tuple

from dataset.continuous import AttributeName, AttributeValue, Attributes, Example

Weights = Dict[AttributeName, float]

def perceptron_step(
        r: float,
        label: AttributeName,
        weights: Weights,
        bias: float,
        example: Example
) -> Tuple[Weights, float]:
    if predict(weights, bias, example) != example[label]:
        change = 1.0 if example[label] == 1 else -1.0
        for attribute in weights:
            weights[attribute] += r * change * example[attribute]
        bias += r * change * 1.0
    return weights, bias

def perceptron_epoch(
        r: float,
        label: AttributeName,
        examples: Iterable[Example],
        weights: Weights,
        bias: float
) -> Tuple[Weights, float]:
    for example in examples:
        weights, bias = perceptron_step(r, label, weights, bias, example)
    return weights, bias

def perceptron(
        epochs: int,
        r: float,
        label: AttributeName,
        attributes: Attributes,
        examples: List[Example]
) -> Tuple[Weights, float]:
    weights = { k: 0.0 for k in attributes }
    bias = 0.0
    for _ in range(epochs):
        shuffle(examples)
        weights, bias = perceptron_epoch(r, label, examples, weights, bias)
    return weights, bias

def predict(weights: Weights, bias: float, example: Example) -> AttributeValue:
    return 1 if (sum(map(lambda attribute: weights[attribute] * example[attribute], weights.keys())) + bias) >= 0 else 0
