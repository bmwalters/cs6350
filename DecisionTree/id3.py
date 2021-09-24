#!/usr/bin/env python3

from typing import Set, Callable
from math import log2

from ..dataset.dataset import Examples, Attribute, AttributeName, partition, most_common_label_value
from .decision_tree import Node, TreeNode, LeafNode

##################
# ID3

EntropyFunc = Callable[[Examples, AttributeName], float]
BestAttributeFunc = Callable[[Examples, Set[Attribute], Attribute], Attribute]

def gain(entropy_func: EntropyFunc) -> BestAttributeFunc:
    def best_attribute(S: Examples, attributes: Set[Attribute], label: Attribute) -> Attribute:
        initial_entropy = entropy_func(S, label.name)

        def gain(S: Examples, A: Attribute):
            total_entropy = 0
            for v in A.values:
                Sv = partition(S, A.name, v)
                if len(Sv) > 0:
                    weight = len(Sv) / len(S)
                    total_entropy += weight * entropy_func(Sv, label.name)
            return initial_entropy - total_entropy

        attributes_iter = iter(attributes)

        best_attribute = attributes_iter.__next__()
        best_gain = gain(S, best_attribute)
        for A in attributes_iter:
            g = gain(S, A)
            if best_gain is None or g > best_gain:
                best_attribute = A
                best_gain = g
        return best_attribute

    return best_attribute

def ID3(best_attribute: BestAttributeFunc, max_depth: int, S: Examples, attributes: Set[Attribute], label: Attribute, depth: int = 0) -> Node:
    assert len(S) > 0

    label_values = set(map(lambda s: s[label.name], S))
    if len(label_values) < 2:
        return LeafNode(label_values.pop())

    if len(attributes) == 0 or depth >= max_depth:
        return LeafNode(most_common_label_value(S, label.name))

    A = best_attribute(S, attributes, label)

    root = TreeNode(A.name)

    for value in A.values:
        S_v = partition(S, A.name, value)
        if len(S_v) == 0:
            root.add_child(value, LeafNode(most_common_label_value(S, label.name)))
        else:
            root.add_child(value, ID3(best_attribute, max_depth, S_v, attributes.difference(set([A])), label, depth + 1))

    return root

#############################
# Entropy/Purity Calculations

def entropy(S: Examples, label: AttributeName) -> float:
    counts = {}
    for s in S:
        if s[label] in counts:
            counts[s[label]] += 1
        else:
            counts[s[label]] = 1
    return sum(map(lambda v: -(v/len(S)) * log2(v/len(S)), counts.values()))

def majority_error(S: Examples, label: AttributeName) -> float:
    counts = { S[0][label]: 1 }
    most_common_count = 1
    for s in S:
        if s[label] in counts:
            counts[s[label]] += 1
        else:
            counts[s[label]] = 1
        if counts[s[label]] > most_common_count:
            most_common_count = counts[s[label]]
    return most_common_count / len(S)

def gini_index(S: Examples, label: AttributeName) -> float:
    counts = {}
    for s in S:
        if s[label] in counts:
            counts[s[label]] += 1
        else:
            counts[s[label]] = 1
    return 1.0 - sum(map(lambda v: (v/len(S)) ** 2, counts.values()))
