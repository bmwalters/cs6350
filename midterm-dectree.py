#!/usr/bin/env python3

import csv

from dataset.dataset import evaluate
from dataset.income import load as load_income_dataset
from DecisionTree.decision_tree import height, predict as predict_decisiontree
from DecisionTree.id3 import ID3, entropy

def main():
    dataset = load_income_dataset("./data/income")

    for max_depth in range(5, 13): # I have observed depth 12 = fully expanded.
        tree = ID3(entropy, max_depth, dataset.train, dataset.train_weights, dataset.attributes, dataset.label)

        predictor = lambda example: predict_decisiontree(tree, example)
        train_error = 1 - evaluate(predictor, dataset.train, dataset.train_weights, dataset.label)
        print(f"max depth {max_depth} real height {height(tree)} train error: {train_error:.2f}")

        def test_predictor(example):
            original_example, fractional_examples = example
            return 1 if sum(map(lambda fe: fe["weight"] * (1 if predictor(fe) == "1" else -1), fractional_examples)) > 0 else 0

        with open(f"generated/test-prediction-{max_depth}", "w") as f:
            writer = csv.writer(f)
            writer.writerow(("ID", "Prediction"))
            writer.writerows(map(lambda ie: (ie[0], test_predictor(ie[1])), enumerate(dataset.test, 1)))

if __name__ == "__main__": pass
if __name__ == "__main__":
    main()
