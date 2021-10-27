#!/usr/bin/env python3

from collections import defaultdict
import csv
from typing import Any, Dict, Iterable, List, Tuple
import os.path

from dataset.dataset import Attribute, AttributeName, Dataset, Example, Weights
from gradient_descent import sgd, compute_loss_gradient

attributes = {
    "age": set(("<25", "25-35", "36-45", "46-55", "56-65", ">65")),
    "workclass": set(("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked")),
    "education.num": set(("<5", "5-7", "8-9", "10", "11-12", "13", "14", ">14")),
    "marital.status": set(("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse")),
    "occupation": set(("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces")),
    "race": set(("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
    "sex": set(("Male", "Female")),
    "capital.gain": set(("<1", "1-2000", "2001-5000", "5001-15000", ">15000")),
    "capital.loss": set(("<1", "1-2000", ">2000")),
    "hours.per.week": set(("<20", "20-29", "30-39", "40-45", "46-59", ">59")),
    "native.country": set(("US-Main", "Caribbean", "Central/South America", "Commonwealth", "Europe", "Asia/ME", "SEA")),
}

label = Attribute("income>50K", set(("0", "1")))

def make_str_bucket(b: Dict[str, Iterable[str]]):
    mapping = {}
    for value, keys in b.items():
        for k in keys:
            mapping[k] = value
    return mapping

bucket = {
    "age": (int, (25, 35, 45, 55, 65)),
    "education.num": (int, (5, 7, 9, 10, 12, 13, 14)),
    "capital.gain": (int, (1, 2000, 5000, 15000)),
    "capital.loss": (int, (1, 2000)),
    "hours.per.week": (int, (20, 29, 39, 45, 59)),
    "native.country": (str, make_str_bucket({
        "US-Main": ("United-States", "Outlying-US(Guam-USVI-etc)"),
        "Caribbean": ("Cuba", "Dominican-Republic", "Haiti", "Jamaica", "Puerto-Rico", "South", "Trinadad&Tobago"),
        "Central/South America": ("Columbia", "Ecuador", "El-Salvador", "Guatemala", "Honduras", "Mexico", "Nicaragua", "Peru"),
        "Commonwealth": ("Canada", "England", "Ireland", "Scotland"),
        "Europe": ("France", "Germany", "Greece", "Holand-Netherlands", "Hungary", "Italy", "Poland", "Portugal", "Yugoslavia"),
        "Asia/ME": ("China", "Hong", "Iran", "Japan", "Taiwan"),
        "SEA": ("Cambodia", "India", "Laos", "Philippines", "Thailand", "Vietnam"),
    }))
}

unknowables = set(("workclass", "occupation", "native.country"))

def make_fes(example, attribute_name: str, unknowable_counts: Dict[AttributeName, int]):
    out = []
    total_count = sum(unknowable_counts.values())
    for possible_value, count in unknowable_counts.items():
        cur_weight = float(example["weight"])
        new_example = example.copy()
        new_example[attribute_name] = possible_value
        new_example["weight"] = (count / total_count) * cur_weight
        out.append(new_example)
    return out

def load_income_dataset(path: str) -> Tuple[Dataset, Weights]:
    train: List[Dict[AttributeName, Any]] = []
    with open(os.path.join(path, "train_final.csv"), "r") as f:
        train = list(csv.DictReader(f))

    # process unknowns
    unknowable_counts: defaultdict[AttributeName, defaultdict[AttributeName, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for example in train:
        example["weight"] = 1.0
        for unknowable in unknowables:
            if example[unknowable] != "?":
                unknowable_counts[unknowable][example[unknowable]] += 1
    fractionalized_train = []
    for example in train:
        fractional_examples = [example]
        unknown = []
        for unknowable in unknowables:
            if example[unknowable] == "?":
                unknown.append(unknowable)
        while len(unknown) > 0:
            to_adj = unknown.pop()
            fractional_examples = [fe for example in fractional_examples for fe in make_fes(example, to_adj, unknowable_counts[to_adj])]
        assert (sum(map(lambda e: e["weight"], fractional_examples)) - 1) < 0.0001
        fractionalized_train += fractional_examples
    train = fractionalized_train

    # process bucketing
    for attr_to_bucket, bucket_params in bucket.items():
        for example in train:
            if bucket_params[0] == int:
                cur_value = int(example[attr_to_bucket])
                buckets = bucket_params[1]
                i = 0
                while i < len(buckets) and cur_value > buckets[i]:
                    i += 1
                new_value = None
                if i == 0:
                    new_value = f"<{buckets[0]}"
                elif i == len(buckets):
                    new_value = f">{buckets[i - 1]}"
                else:
                    lower = buckets[i-1] if i == 1 else buckets[i-1]+1
                    upper = buckets[i]
                    new_value = str(lower) if upper == lower else f"{lower}-{upper}"
                assert new_value in attributes[attr_to_bucket]
                example[attr_to_bucket] = new_value
            elif bucket_params[0] == str:
                cur_value = example[attr_to_bucket]
                new_value = bucket_params[1][cur_value]
                assert new_value in attributes[attr_to_bucket]
                example[attr_to_bucket] = new_value
            else:
                assert False

    weights: List[float] = list(map(lambda e: e["weight"], train))
    return Dataset(train, [], set(map(lambda kv: Attribute(*kv), attributes.items())), label), weights

def main():
    dataset, weights = load_income_dataset("./data/income")
    i = 0
    for t, w in zip(dataset.train, weights):
        if w != 1.0:
            print(t, w)
            i += 1
        if i == 6:
            break
    exit(0)

    iteration_limit = 100_000
    r = 0.125 / 8
    costs, w, bias = None, None, None
    while r > (1 / 1024):
        try:
            costs, w, bias = sgd(
                dataset.train, dataset.attributes, dataset.label,
                r, iteration_limit
            )
        except Exception as e:
            if "diverges" in str(e) or "Result too large" in str(e):
                print("diverged with r", r)
                r /= 2
                continue
            else:
                raise e

        print("r", r, "w", w, "bias", bias)
        print("costs tsv:")
        for i, cost in enumerate(costs, 1):
            print(f"{i}\t{cost}")

        test_cost, _, _ = compute_loss_gradient(w, bias, dataset.test, dataset.label)
        print("test cost", test_cost)

if __name__ == "__main__":
    main()
