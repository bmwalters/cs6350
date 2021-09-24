#!/usr/bin/env python
import os
import os.path
import csv
from statistics import median

from dataset.dataset import Attribute, Dataset
from DecisionTree.decision_tree import evaluate, TreeNode, LeafNode
from DecisionTree.id3 import gain, entropy, ID3, gini_index, majority_error

########
# Config

part_b = True

overwrite = False

##############
# Load Dataset

dataset_path = "./data/bank/"

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

train = None
with open(os.path.join(dataset_path, "train.csv"), "r") as f:
    reader = csv.DictReader(f, list(map(lambda c: c[0], columns)))
    train = list(reader)

numeric_values = {}
unknowable_counts = {}

for example in train:
    for column in columns:
        if column[1] == int:
            value = int(example[column[0]])
            if column[0] in numeric_values:
                numeric_values[column[0]].append(value)
            else:
                numeric_values[column[0]] = [value]
        elif "unknown" in column[1]:
            value = example[column[0]]
            if value != "unknown":
                if column[0] in unknowable_counts:
                    if value in unknowable_counts[column[0]]:
                        unknowable_counts[column[0]][value] += 1
                    else:
                        unknowable_counts[column[0]][value] = 1
                else:
                    unknowable_counts[column[0]] = { value: 1 }

media = { k: median(v) for k, v in numeric_values.items() }

unknowable = {}
for k, counts in unknowable_counts.items():
    counts_iter = iter(counts.items())
    [best_value, best_count] = counts_iter.__next__()
    for value, count in counts_iter:
        if count > best_count:
            best_value = value
            best_count = count
    unknowable[k] = best_value

def fixup_examples(examples):
    for example in examples:
        for k, v in media.items():
            example[k] = "<" if int(example[k]) < v else ">="
        if part_b:
            for k, v in unknowable.items():
                if example[k] == "unknown":
                    example[k] = v

test = None
with open(os.path.join(dataset_path, "test.csv"), "r") as f:
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

dataset = Dataset(train, test, attributes, label_attribute)

########################
# Training and Inference

trees = {}

for (entropy_func, func_name) in [(entropy, "entropy"), (majority_error, "me"), (gini_index, "gi")]:
    for max_depth in range(1, 17):
        part = "_b" if part_b else ""
        filename = f"generated/bank{part}_id3_{func_name}_depth{max_depth}"

        if (not overwrite) and os.path.exists(filename):
            with open(filename, "r") as f:
                trees[filename] = eval(f.read())
            continue

        best_attribute = gain(entropy_func)
        tree = ID3(best_attribute, max_depth, dataset.train, dataset.attributes, dataset.label)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(repr(tree))

        trees[filename] = tree
        print(f"wrote {filename}")

all_examples = dataset.train + dataset.test
for name, tree in trees.items():
    train_error = 1 - evaluate(tree, dataset.train, dataset.label)
    test_error = 1 - evaluate(tree, dataset.test, dataset.label)
    print(f"{name} train error: {train_error:.2f} test error: {test_error:.2f}")
