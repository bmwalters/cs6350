#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Set

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
