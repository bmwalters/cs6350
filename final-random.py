#!/usr/bin/env python3

import csv
import random

from dataset.income import load as load_income_dataset

def main():
    random.seed(4242)
    dataset = load_income_dataset("./data/income")

    # make and save predictions
    with open("generated/final-random.csv", "w") as f:
         writer = csv.writer(f)
         writer.writerow(("ID", "Prediction"))
         writer.writerows(enumerate(map(lambda _: random.random() * 2 - 1, dataset.test), 1))

if __name__ == "__main__": pass
if __name__ == "__main__": main()
