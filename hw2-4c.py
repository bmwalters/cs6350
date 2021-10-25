#!/usr/bin/env python3

import numpy as np

from dataset.concrete import load as load_concrete_dataset

def main():
    dataset = load_concrete_dataset("./data/concrete")

    sorted_attrs = [
        "Cement",
        "Slag",
        "Fly ash",
        "Water",
        "SP",
        "Coarse Aggr",
        "Fine Aggr"
    ]

    X = np.array(list(map(lambda e: [1.0] + [e[a] for a in sorted_attrs], dataset.train))).transpose()
    print(X)
    yraw = [[e[dataset.label]] for e in dataset.train]
    print(yraw)
    Y = np.array(yraw)
    print(X.shape, Y.shape)

    print(np.linalg.inv(np.matmul(X, X.transpose())))
    print(np.matmul(X, Y))

    wstar = np.matmul(np.linalg.inv(np.matmul(X, X.transpose())), np.matmul(X, Y))
    wstr = "[" + " ".join(map(lambda e: f"{e[0]:.4f}", wstar[1:])) + "]"
    print(f"w = {wstr} b = {wstar[0][0]:.4f}")

if __name__ == "__main__":
    main()
