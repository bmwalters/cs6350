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
) -> Tuple[bool, Weights, float]:
    correct = predict(weights, bias, example) == example[label]
    if not correct:
        change = 1.0 if example[label] == 1 else -1.0
        for attribute in weights:
            weights[attribute] += r * change * example[attribute]
        bias += r * change * 1.0
    return correct, weights, bias

def perceptron_epoch(
        r: float,
        label: AttributeName,
        examples: Iterable[Example],
        weights: Weights,
        bias: float
) -> Tuple[int, Weights, float]:
    count_correct = 0
    for example in examples:
        correct, weights, bias = perceptron_step(r, label, weights, bias, example)
        count_correct += 1 if correct else 0
    return count_correct, weights, bias

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
        count_correct, weights, bias = perceptron_epoch(r, label, examples, weights, bias)
        if count_correct == len(examples):
            break
    return weights, bias

def predict(weights: Weights, bias: float, example: Example) -> AttributeValue:
    return 1 if (sum(map(lambda attribute: weights[attribute] * example[attribute], weights.keys())) + bias) >= 0 else 0
