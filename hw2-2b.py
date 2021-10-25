#!/usr/bin/env python3

from collections import defaultdict
from itertools import repeat
import multiprocessing
import os.path
import random
from typing import Iterable, List, Tuple

from dataset.bank import load as load_bank_dataset
from dataset.dataset import AttributeValue, Dataset, evaluate
from DecisionTree.decision_tree import Node, LeafNode, TreeNode, predict as predict_decisiontree
from DecisionTree.id3 import ID3, entropy

def make_classifier(dataset: Dataset):
    bootstrap = random.choices(dataset.train, k=len(dataset.train))
    tree = ID3(entropy, None, bootstrap, repeat(1), dataset.attributes, dataset.label)
    return tree

save_path = "./generated/bagged-trees"

def save_state(trees: List[Node]):
    with open(save_path, "w") as f:
        f.write(repr(trees))

def load_state(overwrite: bool = False) -> List[Node]:
    if overwrite:
        return []
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            return eval(f.read())
    else:
        return []

def evaluate_bagged(T: int, trees: Iterable[Node], dataset: Dataset) -> Tuple[int, float, float]:
    def predictor(example):
        votes: defaultdict[AttributeValue, int] = defaultdict(lambda: 0)
        for tree in trees:
            vote = predict_decisiontree(tree, example)
            votes[vote] += 1
        return max(votes.keys(), key=lambda k: votes[k])

    train_error = 1 - evaluate(predictor, dataset.train, repeat(1), dataset.label)
    test_error = 1 - evaluate(predictor, dataset.test, repeat(1), dataset.label)

    return T, train_error, test_error

def main():
    with multiprocessing.Pool() as pool:
        dataset = load_bank_dataset("./data/bank")
        trees = []
        waitables = []

        print("iteration\ttrain error\ttest error")

        def handle_evaluation(evaluation: Tuple[int, float, float]):
            T, train_error, test_error = evaluation
            print(f"{T}\t{train_error}\t{test_error}")

        def handle_error(error):
            print("*** error", error)

        def handle_tree_created(tree: Node):
            trees.append(tree)
            print("tree created", len(trees))
            task = pool.apply_async(
                evaluate_bagged, (len(trees), trees, dataset),
                callback=handle_evaluation, error_callback=handle_error
            )
            waitables.append(task)

        for _ in range(1, 501):
            task = pool.apply_async(
                make_classifier, (dataset,),
                callback=handle_tree_created,
                error_callback=handle_error
            )
            waitables.append(task)

        for task in waitables:
            task.wait()

if __name__ == "__main__":
    main()
