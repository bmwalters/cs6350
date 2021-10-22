#!/usr/bin/env python

import os
import os.path
import csv
from itertools import islice, repeat
from statistics import median
from typing import List, Tuple

from dataset.dataset import Attribute, Dataset, Examples, Weights, evaluate
from DecisionTree.decision_tree import Node, TreeNode, LeafNode, predict as predict_decisiontree
from DecisionTree.id3 import ID3, entropy
from adaboost import adaboost_step, predict as predict_adaboost

##############
# Load Dataset

dataset_path = "./data/bank/"

columns = [
    ("age", int),
    ("job", [
        "admin.","unknown","unemployed","management","housemaid","entrepreneur",
        "student","blue-collar","self-employed","retired","technician","services"
    ]),
    ("marital", ["married","divorced","single"]),
    ("education", ["unknown","secondary","primary","tertiary"]),
    ("default", ["yes","no"]),
    ("balance", int),
    ("housing", ["yes","no"]),
    ("loan", ["yes","no"]),
    ("contact", ["unknown","telephone","cellular"]),
    ("day", int),
    ("month", [
        "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
    ]),
    ("duration", int),
    ("campaign", int),
    ("pdays", int),
    ("previous", int),
    ("poutcome", ["unknown","other","failure","success"]),
    ("y", ["yes","no"]),
]

train = None
with open(os.path.join(dataset_path, "train.csv"), "r") as f:
    reader = csv.DictReader(f, list(map(lambda c: c[0], columns)))
    train = list(reader)

numeric_values = {}

for example in train:
    for column in columns:
        if column[1] == int:
            value = int(example[column[0]])
            if column[0] in numeric_values:
                numeric_values[column[0]].append(value)
            else:
                numeric_values[column[0]] = [value]

media = { k: median(v) for k, v in numeric_values.items() }

def fixup_examples(examples):
    for example in examples:
        for k, v in media.items():
            example[k] = "<" if int(example[k]) < v else ">="

test = None
with open(os.path.join(dataset_path, "test.csv"), "r") as f:
    reader = csv.DictReader(f, list(map(lambda c: c[0], columns)))
    test = list(reader)

fixup_examples(train)
fixup_examples(test)

def to_attribute(column):
    if column[1] == int:
        return Attribute(column[0], set(["<", ">="]))
    else:
        return Attribute(column[0], set(column[1]))
attributes = set(map(to_attribute, columns[:-1]))
label_attribute = to_attribute(columns[-1])

dataset = Dataset(train, test, attributes, label_attribute)

########################
# Training and Inference

H: List[Tuple[float, Node]] = []
avg_stump_errors = []

D = [1 / len(train) for _ in train]

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
    predictor = predict_adaboost(islice(H, i + 1), predict_decisiontree)
    train_error = 1 - evaluate(predictor, dataset.train, equal_weight_train, dataset.label)
    test_error = 1 - evaluate(predictor, dataset.test, equal_weight_test, dataset.label)
    print(f"{T}\t{train_error:.2f}\t{test_error:.2f}\t{avg_stump_error:.2f}")
