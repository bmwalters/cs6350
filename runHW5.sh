#!/usr/bin/env bash
set -euo pipefail

python3 -m unittest

echo ========
echo HW5 - Q2
echo ========
python3 hw5-2.py

echo ===================
echo HW5 - Q2e (PyTorch)
echo ===================
python3 hw5-pytorch.py
