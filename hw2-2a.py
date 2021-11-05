#!/usr/bin/env python

from itertools import repeat
from typing import List, Tuple

from dataset.bank import load as load_bank_dataset
from dataset.dataset import Attribute, Dataset, Examples, Weights, evaluate
from DecisionTree.decision_tree import Node, TreeNode, LeafNode, predict as predict_decisiontree
from DecisionTree.id3 import ID3, entropy
from EnsembleLearning.adaboost import adaboost_step, predict as predict_adaboost

dataset = load_bank_dataset("./data/bank")

########################
# Training and Inference

H: List[Tuple[float, Node]] = []
avg_stump_errors = []

D = [1 / len(dataset.train) for _ in dataset.train]

for T in range(1, 501):
    sum_stump_error = 0
    num_stumps = 0

    def find_classifier(examples: Examples, weights: Weights) -> Tuple[Node, float]:
        global sum_stump_error, num_stumps

        stump = ID3(entropy, 1, examples, weights, dataset.attributes, dataset.label)
        error = 1 - evaluate(lambda e: predict_decisiontree(stump, e), examples, weights, dataset.label)
        assert error < 0.5

        sum_stump_error += error
        num_stumps += 1

        return stump, error

    alpha, classifier = adaboost_step(dataset.train, dataset.label.name, find_classifier, predict_decisiontree, D)

    average_stump_error = sum_stump_error / num_stumps

    H.append((alpha, classifier))
    avg_stump_errors.append(average_stump_error)

    print(f"completed iteration {T}")

all_examples = dataset.train + dataset.test
equal_weight_train = repeat(1 / len(dataset.train))
equal_weight_test = repeat(1 / len(dataset.test))

print("iterations\ttrain error\ttest error\tavg stump error")
for i in range(len(H)):
    T = i + 1
    avg_stump_error = avg_stump_errors[i - 1]
    predictor = predict_adaboost(H[:i + 1], predict_decisiontree)
    train_error = 1 - evaluate(predictor, dataset.train, equal_weight_train, dataset.label)
    test_error = 1 - evaluate(predictor, dataset.test, equal_weight_test, dataset.label)
    print(f"{T}\t{train_error:.2f}\t{test_error:.2f}\t{avg_stump_error:.2f}")
