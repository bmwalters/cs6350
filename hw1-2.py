#!/usr/bin/env python

import os
import os.path

from dataset.parse import parse_dataset
from DecisionTree.decision_tree import evaluate, TreeNode, LeafNode
from DecisionTree.id3 import gain, entropy, ID3, gini_index, majority_error

car_dataset = parse_dataset("./data/car/")

overwrite = False

trees = {}

for (entropy_func, func_name) in [(entropy, "entropy"), (majority_error, "me"), (gini_index, "gi")]:
    for max_depth in [1, 2, 3, 4, 5, 6]:
        filename = f"generated/car_id3_{func_name}_depth{max_depth}"

        if (not overwrite) and os.path.exists(filename):
            with open(filename, "r") as f:
                trees[filename] = eval(f.read())
            continue

        best_attribute = gain(entropy_func)
        tree = ID3(best_attribute, max_depth, car_dataset.train, car_dataset.attributes, car_dataset.label)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(repr(tree))

        trees[filename] = tree
        print(f"wrote {filename}")

all_examples = car_dataset.train + car_dataset.test
for name, tree in trees.items():
    train_error = 1 - evaluate(tree, car_dataset.train, car_dataset.label)
    test_error = 1 - evaluate(tree, car_dataset.test, car_dataset.label)
    print(f"{name} train error: {train_error:.2f} test error: {test_error:.2f}")
