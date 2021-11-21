#!/usr/bin/env python3

import numpy as np
from scipy.optimize import Bounds, minimize
from typing import List, Tuple

from dataset.continuous import AttributeName, Examples
from .svm import Weights

def svm(C: float, attributes: List[AttributeName], label: AttributeName, examples: Examples):
    X = np.array(list(map(lambda e: list(map(lambda a: e[a], attributes)), examples)))
    Y = np.array(list(map(lambda e: 1 if e[label] == 1 else -1, examples)))

    YY = np.outer(Y, Y)
    XXt = np.dot(X, X.T)

    def objective(alphas):
        AA = np.outer(alphas, alphas)
        return 0.5 * np.sum(AA * YY * XXt) - np.sum(alphas)

    sum_ai_yi_eq0 = {
        "type": "eq",
        "fun": lambda alphas: np.dot(alphas, Y)
    }

    x0 = [0] * len(examples)
    result = minimize(
        fun=objective, x0=x0, method="SLSQP",
        constraints=[sum_ai_yi_eq0], bounds=Bounds(0, C)
    )

    alphas = None
    if result.success:
        alphas = result.x
    else:
        raise result.message

    w = np.dot(X.T, np.multiply(alphas, Y))
    b = np.mean(Y - np.dot(w, X.T))

    return { name: weight for name, weight in zip(attributes, w) }, b
