#!/usr/bin/env python3

from typing import Callable, Set, Dict, List, Tuple, Iterable
from collections import defaultdict
from dataclasses import dataclass, field

##################
# Type Definitions

AttributeName = str
AttributeValue = str

@dataclass(frozen = True)
class Attribute:
    name: AttributeName
    values: Set[AttributeValue] = field(compare=False)

Example = Dict[AttributeName, AttributeValue]
Examples = List[Example]
Weights = Iterable[float]

@dataclass(frozen = True)
class Dataset:
    train: Examples
    test: Examples
    attributes: Set[Attribute]
    label: Attribute

##################
# Dataset Helpers

def partition(S: Examples, weights: Weights, A: AttributeName, v: AttributeValue) -> Tuple[Examples, Weights]:
    sv = []
    wv = []
    for s, weight in zip(S, weights):
        if s[A] == v:
            sv.append(s)
            wv.append(weight)
    return sv, wv

def most_common_label_value(S: Examples, weights: Weights, label: AttributeName) -> AttributeValue:
    counts: defaultdict[AttributeValue, float] = defaultdict(lambda: 0)
    most_common_value = None
    most_common_count = None
    for s, weight in zip(S, weights):
        counts[s[label]] += weight
        if most_common_count is None or counts[s[label]] > most_common_count:
            most_common_value = s[label]
            most_common_count = counts[s[label]]
    assert most_common_value is not None
    return most_common_value

def evaluate(predictor: Callable[[Example], AttributeValue], examples: Examples, weights: Weights, label: Attribute) -> float:
    correct = 0
    total = 0
    for example, weight in zip(examples, weights):
        if predictor(example) == example[label.name]:
            correct += weight
        total += weight
    return correct / total
