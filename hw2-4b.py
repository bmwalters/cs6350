#!/usr/bin/env python3

from dataset.concrete import load as load_concrete_dataset
from gradient_descent import sgd, compute_loss_gradient

def main():
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

        print("r", r, "w", w, "bias", bias)
        print("costs tsv:")
        for i, cost in enumerate(costs, 1):
            print(f"{i}\t{cost}")

        test_cost, _, _ = compute_loss_gradient(w, bias, dataset.test, dataset.label)
        print("test cost", test_cost)

if __name__ == "__main__":
    main()
