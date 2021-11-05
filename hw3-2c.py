#!/usr/bin/env python3

import random

from dataset.continuous import Example, evaluate
from dataset.bank_note import load as load_bank_note_dataset
from average_perceptron import perceptron, predict

def main():
    random.seed(4242)
    dataset = load_bank_note_dataset("./data/bank-note")
    weights, bias = perceptron(10, 1, dataset.label, dataset.attributes, dataset.train)
    print("weights", weights, "bias", bias)

    def predictor(example: Example):
        return predict(weights, bias, example)
    test_error = 1 - evaluate(predictor, dataset.label, dataset.test)
    print("test error", test_error)

if __name__ == "__main__": pass
if __name__ == "__main__": main()
