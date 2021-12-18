#!/usr/bin/env python3

import csv
import os.path
from random import gauss
from typing import Callable

from dataset.continuous import Example as ContinuousExample, evaluate
from dataset.dataset import Example as DiscreteExample
from dataset.income import load as load_income_dataset
from NeuralNetwork.backpropagation import train_sgd, predict

def main():
    dataset = load_income_dataset("./data/income")

    # embed discrete values in examples
    def map_example(example: DiscreteExample) -> ContinuousExample:
        try:
            new_example = {}
            for attribute in dataset.attributes:
                for attribute_value in attribute.values:
                    new_example[f"has_{attribute.name}_{attribute_value}"] = \
                        1.0 if example[attribute.name] == attribute_value else -1.0
            if dataset.label.name in example:
                new_example[dataset.label.name] = 1 if example[dataset.label.name] == "1" else 0
            return new_example
        except Exception as e:
            print(e)
            print(example)
            exit(0)
    train_continuous = list(map(map_example, dataset.train))
    test_continuous = list(map(lambda e: map_example(e[0]), dataset.test))

    # order new attributes
    ordered_attributes = list(test_continuous[0].keys())

    # train net
    T = 5
    normal = lambda: gauss(mu=0, sigma=1)

    net_w = []
    net_path = f"generated/final-ann-discrete-{T}"
    weights_path = f"{net_path}-weights.py"
    output_path = f"{net_path}.csv"
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            net_w = eval(f.read())
    else:
        net_w = train_sgd(
            train_continuous, ordered_attributes, dataset.label.name,
            width=5, T=T, r0=0.1, d=1, initial_weight=normal
        )
    predictor: Callable[[ContinuousExample], float] = \
        lambda example: predict(net_w, example, ordered_attributes)
    binary_predictor = lambda example: 1 if predictor(example) >= 0 else 0

    # evaluate training error
    print("train error", 1.0 - evaluate(binary_predictor, dataset.label.name, train_continuous))

    # save net
    with open(weights_path, "w") as f:
        f.write(repr(net_w))

    # make and save predictions
    with open(output_path, "w") as f:
         writer = csv.writer(f)
         writer.writerow(("ID", "Prediction"))
         writer.writerows(enumerate(map(predictor, test_continuous), 1))

if __name__ == "__main__": pass
if __name__ == "__main__": main()
