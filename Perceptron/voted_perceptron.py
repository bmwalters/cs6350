#!/usr/bin/env python3

from random import shuffle
from typing import Iterable, List, Tuple

from dataset.continuous import AttributeName, AttributeValue, Attributes, Example
from .perceptron import Weights, predict as predict_perceptron

def perceptron_step(
        r: float,
        label: AttributeName,
        weights: List[Weights],
        biases: List[float],
        counts: List[int],
        example: Example
) -> None:
    correct = predict_perceptron(weights[-1], biases[-1], example) == example[label]
    if correct:
        counts[-1] += 1
    else:
        change = 1.0 if example[label] == 1 else -1.0

        new_weights = weights[-1].copy()
        for attribute in new_weights:
            new_weights[attribute] += r * change * example[attribute]
        new_bias = biases[-1] + r * change * 1.0

        weights.append(new_weights)
        biases.append(new_bias)
        counts.append(1)

def perceptron_epoch(
        r: float,
        label: AttributeName,
        examples: Iterable[Example],
        weights: List[Weights],
        biases: List[float],
        counts: List[int]
) -> None:
    for example in examples:
        perceptron_step(r, label, weights, biases, counts, example)

def perceptron(
        epochs: int,
        r: float,
        label: AttributeName,
        attributes: Attributes,
        examples: List[Example]
) -> Tuple[List[Weights], List[float], List[int]]:
    weights = [{ k: 0.0 for k in attributes }]
    biases = [0.0]
    counts = [0]
    for _ in range(epochs):
        shuffle(examples)
        perceptron_epoch(r, label, examples, weights, biases, counts)
        if counts[-1] == len(examples):
            break
    return weights, biases, counts

def predict(weights: List[Weights], biases: List[float], counts: List[int], example: Example) -> AttributeValue:
    def vote(weights, bias, count):
        return count * (1 if predict_perceptron(weights, bias, example) == 1 else -1)
    return 1 if sum(map(lambda wbc: vote(*wbc), zip(weights, biases, counts))) >= 0 else 0
