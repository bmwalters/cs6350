#!/usr/bin/env python3

from typing import Callable, Dict, Optional, Set
from math import log2
from itertools import islice
from collections import defaultdict

from dataset.dataset import Examples, Attribute, AttributeName, AttributeValue, Weights, partition, most_common_label_value
from .decision_tree import Node, TreeNode, LeafNode

##################
# ID3

EntropyFunc = Callable[[Examples, Weights, AttributeName], float]

def attribute_gains(entropy_func: EntropyFunc, S: Examples, weights: Weights, attributes: Set[Attribute], label: Attribute) -> Dict[Attribute, float]:
    initial_entropy = entropy_func(S, weights, label.name)

    def attribute_gain(A: Attribute) -> float:
        total_entropy = 0
        for v in A.values:
            Sv, Wv = partition(S, weights, A.name, v)
            if len(Sv) > 0:
                weight = sum(islice(Wv, len(Sv))) / sum(islice(weights, len(S)))
                total_entropy += weight * entropy_func(Sv, Wv, label.name)
        return initial_entropy - total_entropy

    return dict(map(lambda A: (A, attribute_gain(A)), attributes))

def ID3(entropy_func: EntropyFunc, max_depth: Optional[int], S: Examples, weights: Weights, attributes: Set[Attribute], label: Attribute) -> Node:
    def _ID3(S: Examples, weights: Weights, attributes: Set[Attribute], depth: int = 0) -> Node:
        assert len(S) > 0
        assert len(S) == len(weights) if isinstance(weights, list) else True

        label_values = set(map(lambda s: s[label.name], S))
        if len(label_values) < 2:
            return LeafNode(label_values.pop())

        if len(attributes) == 0 or (max_depth is not None and depth >= max_depth):
            return LeafNode(most_common_label_value(S, weights, label.name))

        gains = attribute_gains(entropy_func, S, weights, attributes, label)
        A = max(gains.keys(), key=lambda name: gains[name])

        root = TreeNode(A.name)

        for value in A.values:
            S_v, W_v = partition(S, weights, A.name, value)
            if len(S_v) == 0:
                root.add_child(value, LeafNode(most_common_label_value(S, weights, label.name)))
            else:
                root.add_child(value, _ID3(S_v, W_v, attributes.difference(set([A])), depth + 1))

        return root

    return _ID3(S, weights, attributes)

#############################
# Entropy/Purity Calculations

def entropy(S: Examples, weights: Weights, label: AttributeName) -> float:
    counts: defaultdict[AttributeValue, float] = defaultdict(lambda: 0)
    total = 0.0
    for s, weight in zip(S, weights):
        counts[s[label]] += weight
        total += weight
    return -sum(map(lambda v: (v/total) * log2(v/total), counts.values()))

def majority_error(S: Examples, weights: Weights, label: AttributeName) -> float:
    counts: defaultdict[AttributeValue, float] = defaultdict(lambda: 0)
    total = 0.0
    most_common_count = None
    for s, weight in zip(S, weights):
        counts[s[label]] += weight
        total += weight
        if most_common_count is None or counts[s[label]] > most_common_count:
            most_common_count = counts[s[label]]
    return 1.0 - most_common_count / total

def gini_index(S: Examples, weights: Weights, label: AttributeName) -> float:
    counts: defaultdict[AttributeValue, float] = defaultdict(lambda: 0)
    total = 0.0
    for s, weight in zip(S, weights):
        counts[s[label]] += weight
        total += weight
    return 1.0 - sum(map(lambda v: (v/total) ** 2, counts.values()))
