#!/usr/bin/env python3

from typing import Set, Dict, List
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

@dataclass(frozen = True)
class Dataset:
    train: Examples
    test: Examples
    attributes: Set[Attribute]
    label: Attribute

##################
# Dataset Helpers

def partition(S: Examples, A: AttributeName, v: AttributeValue) -> Examples:
    return [s for s in S if s[A] == v]

def most_common_label_value(S: Examples, label: AttributeName) -> AttributeValue:
    counts = { S[0][label]: 1 }
    most_common_value = S[0][label]
    most_common_count = 1
    for s in S:
        if s[label] in counts:
            counts[s[label]] += 1
        else:
            counts[s[label]] = 1
        if counts[s[label]] > most_common_count:
            most_common_value = s[label]
            most_common_count = counts[s[label]]
    return most_common_value
