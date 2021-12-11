#!/usr/bin/env python3

import random

from dataset.continuous import Example, evaluate
from dataset.bank_note import attributes, load as load_bank_note_dataset
from NeuralNetwork.backpropagation import train_sgd, predict

def main():
    random.seed(4242)
    dataset = load_bank_note_dataset("./data/bank-note")

    T = 100
    r0 = 0.1
    d = 1

    def trial(width, initial_weight):
        weights = train_sgd(
            dataset.train, attributes, dataset.label,
            width, T, r0, d, initial_weight
        )

        def predictor(example: Example):
            x = [1.0] + list(map(lambda a: example[a], attributes))
            return 1 if predict(weights, x) >= 0 else 0
        train_error = 1 - evaluate(predictor, dataset.label, dataset.train)
        test_error = 1 - evaluate(predictor, dataset.label, dataset.test)
        print(f"train error {train_error} test error {test_error}")

    for width in [5, 10, 25, 50]: # [5, 10, 25, 50, 100]
        print(f"=== Q2b width = {width} ===")
        trial(width, lambda: random.gauss(0, 1))

    print("=== Q2c width = 5, initial_weights = 0 ===")
    trial(5, lambda: 0)

if __name__ == "__main__": pass
if __name__ == "__main__": main()
