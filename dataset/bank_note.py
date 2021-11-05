#!/usr/bin/env python3

import csv
from typing import Any, Dict, List
import os.path

from .continuous import Dataset

attributes = [
    "variance", "skewiness", "curtosis", "entropy"
]
label = "genuine?"

columns = attributes + [label]

def load(path: str) -> Dataset:
    def parse(train_or_test: str) -> List[Dict[str, Any]]:
        with open(os.path.join(path, train_or_test + ".csv"), "r") as f:
            reader = csv.DictReader(f, columns, quoting=csv.QUOTE_NONNUMERIC)
            return list(reader)

    train = parse("train")
    test = parse("test")

    return Dataset(train, test, set(attributes), label)
