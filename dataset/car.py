#!/usr/bin/env python3

import csv
import os.path

from .dataset import Dataset, Attribute

def load(path: str) -> Dataset:
    def parse_attribute_line(line: str) -> Attribute:
        [name, values] = map(lambda s: s.strip().rstrip("."), line.split(":"))
        return Attribute(name, set(values.split(", ")))

    # parse data-desc file
    desc = {}
    with open(os.path.join(path, "data-desc.txt"), "r") as f:
        cur_key = None
        for line in f:
            if line.startswith("|"):
                cur_key = line[1:].strip()
            elif line.strip() == "":
                continue
            else:
                assert cur_key is not None
                if cur_key in desc:
                    desc[cur_key].append(line.strip())
                else:
                    desc[cur_key] = [line.strip()]

    label_name = "label"
    label_values = desc["label values"][0].split(", ")
    label_attribute = Attribute(label_name, set(label_values))

    attributes = set(map(parse_attribute_line, desc["attributes"]))
    columns = desc["columns"][0].split(",")

    train = None
    with open(os.path.join(path, "train.csv"), "r") as f:
        reader = csv.DictReader(f, columns)
        train = list(reader)

    test = None
    with open(os.path.join(path, "test.csv"), "r") as f:
        reader = csv.DictReader(f, columns)
        test = list(reader)

    return Dataset(train, test, attributes, label_attribute)
