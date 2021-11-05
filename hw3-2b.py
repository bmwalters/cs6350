#!/usr/bin/env python3

import random

from dataset.continuous import Example, evaluate
from dataset.bank_note import attributes as bank_note_attributes_ordered, load as load_bank_note_dataset
from Perceptron.voted_perceptron import perceptron, predict as predict_voted_perceptron

def main():
    random.seed(4242)
    dataset = load_bank_note_dataset("./data/bank-note")
    weights, biases, counts = perceptron(10, 1, dataset.label, dataset.attributes, dataset.train)

    print("w_variance,w_skewiness,w_curtosis,w_entropy,bias,count")
    for ws, bias, count in zip(weights, biases, counts):
        print(",".join([*[f"{ws[a]:.2f}" for a in bank_note_attributes_ordered], f"{bias:.2f}", str(count)]))

    def predictor(example: Example):
        return predict_voted_perceptron(weights, biases, counts, example)
    test_error = 1 - evaluate(predictor, dataset.label, dataset.test)
    print("test error", test_error)

if __name__ == "__main__": pass
if __name__ == "__main__": main()
