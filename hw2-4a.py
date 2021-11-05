#!/usr/bin/env python3

from dataset.concrete import load as load_concrete_dataset
from LinearRegression.gradient_descent import bgd, compute_loss_gradient

def main():
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

    print("r", r, "w", w, "bias", bias, "costs", costs)

    test_cost, _, _ = compute_loss_gradient(w, bias, dataset.test, dataset.label)
    print("test cost", test_cost)

if __name__ == "__main__":
    main()
