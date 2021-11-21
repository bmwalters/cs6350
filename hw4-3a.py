#!/usr/bin/env python3

import random

from dataset.continuous import Example, evaluate
from dataset.bank_note import attributes, load as load_bank_note_dataset
from SVM.dual import svm
from SVM.svm import predict

def main():
    random.seed(4242)
    dataset = load_bank_note_dataset("./data/bank-note")

    for C in [100/873, 500/873, 700/873]:
        print(f"C = {C}")
        weights, bias = svm(C, attributes, dataset.label, dataset.train)
        for name in attributes:
            print(f"\t{name} {weights[name]}")
        print(f"\tbias {bias}")

        def predictor(example: Example):
            return 1 if predict(weights, bias, example) >= 0 else 0
        train_error = 1 - evaluate(predictor, dataset.label, dataset.train)
        test_error = 1 - evaluate(predictor, dataset.label, dataset.test)
        print(f"\ttrain error {train_error} test_error {test_error}")

if __name__ == "__main__": pass
if __name__ == "__main__": main()
