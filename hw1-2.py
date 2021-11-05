#!/usr/bin/env python

from itertools import repeat
import os
import os.path

from dataset.dataset import evaluate
from dataset.car import load as load_car_dataset
from DecisionTree.decision_tree import TreeNode, LeafNode, predict
from DecisionTree.id3 import entropy, ID3, gini_index, majority_error

car_dataset = load_car_dataset("./data/car/")

overwrite = False

trees = {}

for (entropy_func, func_name) in [(entropy, "entropy"), (majority_error, "me"), (gini_index, "gi")]:
    for max_depth in [1, 2, 3, 4, 5, 6]:
        filename = f"generated/car_id3_{func_name}_depth{max_depth}"

        if (not overwrite) and os.path.exists(filename):
            with open(filename, "r") as f:
                trees[filename] = eval(f.read())
            continue

        tree = ID3(entropy_func, max_depth, car_dataset.train, repeat(1), car_dataset.attributes, car_dataset.label)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(repr(tree))

        trees[filename] = tree
        print(f"wrote {filename}")

all_examples = car_dataset.train + car_dataset.test
for name, tree in trees.items():
    predictor = lambda example: predict(tree, example)
    train_error = 1 - evaluate(predictor, car_dataset.train, repeat(1), car_dataset.label)
    test_error = 1 - evaluate(predictor, car_dataset.test, repeat(1), car_dataset.label)
    print(f"{name} train error: {train_error:.2f} test error: {test_error:.2f}")
