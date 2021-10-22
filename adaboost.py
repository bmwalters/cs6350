#!/usr/bin/env python3

from collections import defaultdict
import math
from typing import Callable, Iterable, List, Tuple, TypeVar

from dataset.dataset import AttributeName, AttributeValue, Example, Examples, Weights

TClassifier = TypeVar("TClassifier")

# mutates D
def adaboost_step(
        train: Examples,
        label_attribute_name: AttributeName,
        find_classifier: Callable[[Examples, Weights], Tuple[TClassifier, float]],
        predict_base: Callable[[TClassifier, Example], AttributeValue],
        D: List[float]
) -> Tuple[float, TClassifier]:
    # find a classifier better than chance on weighted examples
    h, error = find_classifier(train, D)
    assert error < 0.5

    # compute alpha and store classifier
    alpha = (1 / 2) * math.log((1 - error) / error)

    # update D for t+1
    for i, example in enumerate(train):
        correct = 1 if example[label_attribute_name] == predict_base(h, example) else -1
        D[i] = D[i] * math.exp(-alpha * correct)

    # normalize D
    Z = sum(D)
    for i, di in enumerate(D):
        D[i] = di / Z

    return alpha, h

def adaboost(
        train: Examples,
        label_attribute_name: AttributeName,
        T: int,
        find_classifier: Callable[[Examples, Weights], Tuple[TClassifier, float]],
        predict_base: Callable[[TClassifier, Example], AttributeValue]
) -> List[Tuple[float, TClassifier]]:
    D = [1 / len(train) for _ in train]
    H: List[Tuple[float, TClassifier]] = []

    for _ in range(T):
        alpha, h = adaboost_step(train, label_attribute_name, find_classifier, predict_base, D)
        H.append((alpha, h))

    return H

def predict(
        H: Iterable[Tuple[float, TClassifier]],
        predict_base: Callable[[TClassifier, Example], AttributeValue]
):
    def final_classifier(example: Example):
        votes: defaultdict[AttributeValue, float] = defaultdict(lambda: 0.0)
        for alpha, h in H:
            votes[predict_base(h, example)] += alpha
        return max(votes.keys(), key=lambda k: votes[k])
    return final_classifier
