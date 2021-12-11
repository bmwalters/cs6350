#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import random

def main():
    random.seed(4242)
    def read_csv(path):
        examples = []
        with open(path, "r") as f:
            for line in f:
                values = list(map(float, line.split(",")))
                examples.append((torch.Tensor(values[:-1]), torch.Tensor([-1 if values[-1] == 0 else 1])))
        return examples
    train = read_csv("./data/bank-note/train.csv")
    test = read_csv("./data/bank-note/test.csv")

    T = 10

    for activation in ["tanh", "ReLU"]:
        for depth in [3,5,9]:
            for width in [5,10,25,50,100]:
                print(f"=== activation = {activation} depth = {depth} width = {width} ===")
                make_activation = (lambda: torch.nn.ReLU()) if activation == "ReLU" else (lambda: torch.nn.Tanh())
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, width),
                    *[thing for _ in range(depth - 2) for thing in [make_activation(), torch.nn.Linear(width, width)]],
                    make_activation(),
                    torch.nn.Linear(width, 1)
                )

                def init_weights(m):
                    if isinstance(m, torch.nn.Linear):
                        if activation == "ReLU":
                            torch.nn.init.kaiming_normal_(m.weight)
                        else:
                            torch.nn.init.xavier_uniform_(m.weight)
                        m.bias.data.fill_(0.01)
                model.apply(init_weights)

                loss_fn = torch.nn.MSELoss(reduction="sum")
                optimizer = torch.optim.Adam(model.parameters())

                for _ in range(T):
                    for x, y in train:
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                def count_correct(examples):
                    correct = 0
                    for x, y in examples:
                        y_pred = -1 if model(x)[0] < 0 else 1
                        if y[0] == y_pred:
                            correct += 1
                    return correct
                train_error = 1.0 - count_correct(train) / len(train)
                test_error = 1.0 - count_correct(test) / len(test)
                print("\ttrain error", train_error, "test error", test_error)

if __name__ == "__main__": pass
if __name__ == "__main__": main()
