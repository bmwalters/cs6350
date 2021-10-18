## Overview
This is a machine learning library developed by Bradley Walters for CS6350 at the University of Utah.

## Decision Trees

``` python
from DecisionTree.decision_tree import evaluate
from DecisionTree.id3 import gain, entropy, ID3, gini_index, majority_error

# first construct a dataset with train, test, label, and attributes properties

tree = ID3(best_attribute=gain(majority_error), max_depth=6,
           dataset.train, dataset.attributes, dataset.label)

test_error = 1 - evaluate(tree, dataset.test, dataset.label)
```
