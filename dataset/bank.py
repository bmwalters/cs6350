#!/usr/bin/env python3

import os
import os.path
import csv
from statistics import median

from dataset.dataset import Attribute, Dataset

columns = [
    ("age", int),
    ("job", [
        "admin.","unknown","unemployed","management","housemaid","entrepreneur",
        "student","blue-collar","self-employed","retired","technician","services"
    ]),
    ("marital", ["married","divorced","single"]),
    ("education", ["unknown","secondary","primary","tertiary"]),
    ("default", ["yes","no"]),
    ("balance", int),
    ("housing", ["yes","no"]),
    ("loan", ["yes","no"]),
    ("contact", ["unknown","telephone","cellular"]),
    ("day", int),
    ("month", [
        "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
    ]),
    ("duration", int),
    ("campaign", int),
    ("pdays", int),
    ("previous", int),
    ("poutcome", ["unknown","other","failure","success"]),
    ("y", ["yes","no"]),
]

def load(path: str) -> Dataset:
    train = None
    with open(os.path.join(path, "train.csv"), "r") as f:
        reader = csv.DictReader(f, list(map(lambda c: c[0], columns)))
        train = list(reader)

    numeric_values = {}

    for example in train:
        for column in columns:
            if column[1] == int:
                value = int(example[column[0]])
                if column[0] in numeric_values:
                    numeric_values[column[0]].append(value)
                else:
                    numeric_values[column[0]] = [value]

    media = { k: median(v) for k, v in numeric_values.items() }

    def fixup_examples(examples):
        for example in examples:
            for k, v in media.items():
                example[k] = "<" if int(example[k]) < v else ">="

    test = None
    with open(os.path.join(path, "test.csv"), "r") as f:
        reader = csv.DictReader(f, list(map(lambda c: c[0], columns)))
        test = list(reader)

    fixup_examples(train)
    fixup_examples(test)

    def to_attribute(column):
        if column[1] == int:
            return Attribute(column[0], set(["<", ">="]))
        else:
            return Attribute(column[0], set(column[1]))
    attributes = set(map(to_attribute, columns[:-1]))
    label_attribute = to_attribute(columns[-1])

    return Dataset(train, test, attributes, label_attribute)
