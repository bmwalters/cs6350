#!/usr/bin/env python3

from random import shuffle
from typing import Dict, Iterable, List, Tuple

from dataset.continuous import AttributeName, Attributes, Example
from perceptron import Weights, predict as predict_perceptron

def perceptron_step(
        r: float,
        label: AttributeName,
        weights: Weights,
        bias: float,
        a_weights: Weights,
        a_bias: float,
        example: Example
) -> Tuple[bool, Weights, float, Weights, float]:
    correct = predict_perceptron(weights, bias, example) == example[label]
    if not correct:
        change = 1.0 if example[label] == 1 else -1.0

        for attribute in weights:
            weights[attribute] += r * change * example[attribute]
        bias += r * change * 1.0

    for a in a_weights.keys():
        a_weights[a] += weights[a]
    a_bias += bias

    return correct, weights, bias, a_weights, a_bias

def perceptron_epoch(
        r: float,
        label: AttributeName,
        examples: Iterable[Example],
        weights: Weights,
        bias: float,
        a_weights: Weights,
        a_bias: float
) -> Tuple[int, Weights, float, Weights, float]:
    count_correct = 0
    for example in examples:
        correct, weights, bias, a_weights, a_bias = perceptron_step(r, label, weights, bias, a_weights, a_bias, example)
        count_correct += 1 if correct else 0
    return count_correct, weights, bias, a_weights, a_bias

def perceptron(
        epochs: int,
        r: float,
        label: AttributeName,
        attributes: Attributes,
        examples: List[Example]
) -> Tuple[Weights, float]:
    weights = { k: 0.0 for k in attributes }
    bias = 0.0

    a_weights = weights.copy()
    a_bias = bias

    for _ in range(epochs):
        shuffle(examples)
        count_correct, weights, bias, a_weights, a_bias = perceptron_epoch(r, label, examples, weights, bias, a_weights, a_bias)
        if count_correct == len(examples):
            break

    return a_weights, a_bias

predict = predict_perceptron
