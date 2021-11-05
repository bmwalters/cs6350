#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Callable, Dict, List, Set

AttributeName = str
AttributeValue = float
Attributes = Set[AttributeName]
Example = Dict[AttributeName, AttributeValue]
Examples = List[Example]

@dataclass(frozen = True)
class Dataset:
    train: Examples
    test: Examples
    attributes: Attributes
    label: AttributeName

def evaluate(predictor: Callable[[Example], AttributeValue], label: AttributeName, examples: Examples) -> float:
    correct = 0
    for example in examples:
        if predictor(example) == example[label]:
            correct += 1
    return correct / len(examples)
