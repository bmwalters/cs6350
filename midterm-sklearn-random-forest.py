#!/usr/bin/env python3

import csv
import os.path
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from typing import List, Tuple

from dataset.income import attributes as raw_attributes, bucket as raw_bucket, label, process_bucketing, unknowables

def load_income_dataset(path: str) -> Tuple[List[List[int]], List[int], List[List[int]]]:
    # read files
    train = []
    with open(os.path.join(path, "train_final.csv"), "r") as f:
        train = list(csv.DictReader(f))
    test = []
    with open(os.path.join(path, "test_final.csv"), "r") as f:
        test = list(csv.DictReader(f))

    # temporarily ignore unknown values - will fix later
    attributes = raw_attributes.copy()
    bucket = raw_bucket.copy()
    for unknowable in unknowables:
        attributes[unknowable] = attributes[unknowable].copy().union(set(("?",)))
        if unknowable in bucket:
            assert bucket[unknowable][0] == str # thank goodness this is the case
            str_dict = bucket[unknowable][1].copy()
            str_dict["?"] = "?"
            bucket[unknowable] = (str, str_dict)

    # do bucketing
    process_bucketing(train, attributes, bucket)
    process_bucketing(test, attributes, bucket)

    # convert Dict[str, str]s to List[str]s
    sorted_attributes = sorted(raw_attributes.keys())
    new_train = []
    Y_train = []
    new_test = []
    for example in train:
        new_example = []
        for attr in sorted_attributes:
            new_example.append(example[attr])
        Y_train.append(1 if example[label.name] == "1" else 0)
        new_train.append(new_example)
    for example in test:
        new_example = []
        for attr in sorted_attributes:
            new_example.append(example[attr])
        new_test.append(new_example)
    train = new_train
    test = new_test

    # convert List[str]s to List[int]s
    categories = list(map(lambda k: list(sorted(raw_attributes[k])), sorted_attributes))
    missing = 4242
    enc = OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=missing)
    enc._fit(train + test, handle_unknown=enc.handle_unknown)
    train = enc.transform(train)
    test = enc.transform(test)

    # handle unknown values
    imputer = IterativeImputer(missing_values=missing)
    train = imputer.fit_transform(train)
    test = imputer.fit_transform(test)

    return train, Y_train, test

def main():
    trainX, Y, testX = load_income_dataset("./data/income")

    clf = RandomForestClassifier()
    clf = clf.fit(trainX, Y)

    with open(f"generated/sklearn-random-forest-imputed.csv", "w") as f:
         writer = csv.writer(f)
         writer.writerow(("ID", "Prediction"))
         writer.writerows(enumerate(clf.predict(testX), 1))

if __name__ == "__main__": pass
if __name__ == "__main__":
    main()
