## CS 6350 - Information for TAs
This is a machine learning library developed by Bradley Walters for CS6350 at the University of Utah.

## Decision Trees

``` python
from itertools import repeat

from dataset.dataset import evaluate
from dataset.car import load as load_car_dataset
from DecisionTree.decision_tree import TreeNode, LeafNode, predict
from DecisionTree.id3 import entropy, ID3

car_dataset = load_car_dataset("./data/car/")

tree = ID3(
    entropy, 6, car_dataset.train, repeat(1),
    car_dataset.attributes, car_dataset.label
)

predictor = lambda example: predict(tree, example)
test_error = 1 - evaluate(predictor, car_dataset.test, repeat(1), car_dataset.label)
```

## Adaboost

``` python
from itertools import repeat

from dataset.bank import load as load_bank_dataset
from DecisionTree.decision_tree import predict as predict_decisiontree
from DecisionTree.id3 import ID3, entropy
from EnsembleLearning.adaboost import adaboost, predict

bank_dataset = load_bank_dataset("./data/bank/")

def find_stump(examples, weights):
    stump = ID3(
        entropy, 1, examples, weights,
        bank_dataset.attributes, bank_dataset.label
    )
    error = 1 - evaluate(
        lambda e: predict_decisiontree(stump, e),
        examples, weights, bank_dataset.label
    )
    return stump, error

ensemble = adaboost(
    bank_dataset.train, bank_dataset.label.name, 10,
    find_stump, predict_decisiontree
)

predictor = predict(ensemble, predict_decisiontree)
test_error = 1 - evaluate(predictor, bank_dataset.test, repeat(1), bank_dataset.label)
```

## Bagging
Not yet implemented.

## Random Forest
Not yet implemented.

## LMS with Batch Gradient Descent

``` python
from dataset.concrete import load as load_concrete_dataset
from LinearRegression.gradient_descent import bgd, compute_loss_gradient

dataset = load_concrete_dataset("./data/concrete")

iteration_limit = 1_000_000
r = 1.0
costs, w, bias = None, None, None
while True:
    try:
        costs, w, bias = bgd(
            dataset.train, dataset.attributes, dataset.label,
            r, iteration_limit
        )
        break
    except Exception as e:
        if "diverges" in str(e) or "Result too large" in str(e):
            print("diverged with r", r)
            r /= 2
            continue
        else:
            raise e

test_cost, _, _ = compute_loss_gradient(w, bias, dataset.test, dataset.label)
```

## LMS with Stochastic Gradient Descent

``` python
from dataset.concrete import load as load_concrete_dataset
from LinearRegression.gradient_descent import sgd, compute_loss_gradient

dataset = load_concrete_dataset("./data/concrete")

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

    test_cost, _, _ = compute_loss_gradient(w, bias, dataset.test, dataset.label)
```

## Perceptron

``` python
import random

from dataset.continuous import evaluate
from dataset.bank_note import load as load_bank_note_dataset
from Perceptron.perceptron import perceptron, predict

dataset = load_bank_note_dataset("./data/bank-note")

random.seed(4242)
weights, bias = perceptron(10, 1, dataset.label, dataset.attributes, dataset.train)

predictor = lambda example: predict(weights, bias, example)
test_error = 1 - evaluate(predictor, dataset.label, dataset.test)
```
