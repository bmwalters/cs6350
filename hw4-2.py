#!/usr/bin/env python3

import random

from dataset.continuous import Example, evaluate
from dataset.bank_note import attributes, load as load_bank_note_dataset
from SVM.svm import predict, svm

def main():
    random.seed(4242)
    dataset = load_bank_note_dataset("./data/bank-note")

    def gamma_schedule_2a(t: int):
        gamma0 = 0.1
        alpha = 0.01
        return gamma0 / (1 + (gamma0 / alpha) * t)

    def gamma_schedule_2b(t: int):
        gamma0 = 0.001
        return gamma0 / (1 + t)

    for schedule_name, gamma_schedule in [("2a", gamma_schedule_2a), ("2b", gamma_schedule_2b)]:
        print("schedule", schedule_name)
        for C in [100/873, 500/873, 700/873]:
            print(f"\tC = {C}")
            weights, bias = svm(100, C, gamma_schedule, dataset.attributes, dataset.label, dataset.train)
            for name in attributes:
                print(f"\t\t{name} {weights[name]}")
            print(f"\t\tbias {bias}")

            def predictor(example: Example):
                return 1 if predict(weights, bias, example) >= 0 else 0
            train_error = 1 - evaluate(predictor, dataset.label, dataset.train)
            test_error = 1 - evaluate(predictor, dataset.label, dataset.test)
            print(f"\t\ttrain error {train_error} test_error {test_error}")

if __name__ == "__main__": pass
if __name__ == "__main__": main()
