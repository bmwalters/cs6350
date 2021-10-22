#!/usr/bin/env python3

import unittest

from dataset.dataset import Attribute, partition, evaluate
from DecisionTree.decision_tree import LeafNode, TreeNode, predict
from DecisionTree.id3 import ID3, attribute_gains, majority_error

O = Attribute("O", set(("S", "O", "R")))
T = Attribute("T", set(("H", "M", "C")))
H = Attribute("H", set(("H", "N", "L")))
W = Attribute("W", set(("S", "W")))
attributes = set((O, T, H, W))
label = Attribute("Play?", set(("-", "+")))
examples = [
    { "O": "S", "T": "H", "H": "H", "W": "W", "Play?": "-" },
    { "O": "S", "T": "H", "H": "H", "W": "S", "Play?": "-" },
    { "O": "O", "T": "H", "H": "H", "W": "W", "Play?": "+" },
    { "O": "R", "T": "M", "H": "H", "W": "W", "Play?": "+" },
    { "O": "R", "T": "C", "H": "N", "W": "W", "Play?": "+" },
    { "O": "R", "T": "C", "H": "N", "W": "S", "Play?": "-" },
    { "O": "O", "T": "C", "H": "N", "W": "S", "Play?": "+" },
    { "O": "S", "T": "M", "H": "H", "W": "W", "Play?": "-" },
    { "O": "S", "T": "C", "H": "N", "W": "W", "Play?": "+" },
    { "O": "R", "T": "M", "H": "N", "W": "W", "Play?": "+" },
    { "O": "S", "T": "M", "H": "N", "W": "S", "Play?": "+" },
    { "O": "O", "T": "M", "H": "H", "W": "S", "Play?": "+" },
    { "O": "O", "T": "H", "H": "N", "W": "W", "Play?": "+" },
    { "O": "R", "T": "M", "H": "H", "W": "S", "Play?": "-" },
]
weights = [1 for _ in examples]

class TestTennisClassifier(unittest.TestCase):
    def test_majority_error(self):
        result = majority_error(examples, weights, label.name)
        self.assertAlmostEqual(result, 5/14, places=3)

    def test_gain(self):
        gains = attribute_gains(majority_error, examples, weights, attributes, label)
        self.assertAlmostEqual(gains[O], 1/14, places=3)
        self.assertAlmostEqual(gains[T], 0, places=3)
        self.assertAlmostEqual(gains[H], 1/14, places=3)
        self.assertAlmostEqual(gains[W], 0, places=3)

    def test_id3_depth_1(self):
        tree = ID3(majority_error, 1, examples, weights, attributes, label)

        assert isinstance(tree, TreeNode)

        self.assertIn(tree.attribute_name, (O.name, H.name))

        if tree.attribute_name == O.name:
            childS = tree.children["S"]
            childO = tree.children["O"]
            childR = tree.children["R"]
            assert isinstance(childS, LeafNode)
            assert isinstance(childO, LeafNode)
            assert isinstance(childR, LeafNode)
            self.assertEqual(childS.label, "-")
            self.assertEqual(childO.label, "+")
            self.assertEqual(childR.label, "+")
        elif tree.attribute_name == H.name:
            childH = tree.children["H"]
            childN = tree.children["N"]
            childL = tree.children["L"]
            assert isinstance(childH, LeafNode)
            assert isinstance(childN, LeafNode)
            assert isinstance(childL, LeafNode)
            self.assertEqual(childH.label, "-")
            self.assertEqual(childN.label, "+")
            self.assertEqual(childL.label, "+")

    def test_majority_error_partitioned_O(self):
        SvS, WvS = partition(examples, weights, O.name, "S")
        SvO, WvO = partition(examples, weights, O.name, "O")
        SvR, WvR = partition(examples, weights, O.name, "R")

        self.assertEqual(SvS, [examples[0], examples[1], examples[7], examples[8], examples[10]])
        self.assertEqual(SvO, [examples[2], examples[6], examples[11], examples[12]])
        self.assertEqual(SvR, [examples[3], examples[4], examples[5], examples[9], examples[13]])

        for _, w in zip(range(5), WvS):
            self.assertAlmostEqual(w, 1)
        for _, w in zip(range(4), WvO):
            self.assertAlmostEqual(w, 1)
        for _, w in zip(range(5), WvR):
            self.assertAlmostEqual(w, 1)

        self.assertAlmostEqual(majority_error(SvS, WvS, label.name), 2/5, places=3)
        self.assertAlmostEqual(majority_error(SvO, WvO, label.name), 0, places=3)
        self.assertAlmostEqual(majority_error(SvR, WvR, label.name), 2/5, places=3)

    def test_gains_partitioned_O(self):
        SvS, WvS = partition(examples, weights, O.name, "S")
        gainsS = attribute_gains(majority_error, SvS, WvS, set((T, H, W)), label)
        self.assertAlmostEqual(gainsS[T], 1/5, places=3)
        self.assertAlmostEqual(gainsS[H], 2/5, places=3)
        self.assertGreater(gainsS[H], gainsS[W])

        SvR, WvR = partition(examples, weights, O.name, "R")
        gainsR = attribute_gains(majority_error, SvR, WvR, set((T, H, W)), label)
        self.assertAlmostEqual(gainsR[T], 0, places=3)
        self.assertAlmostEqual(gainsR[H], 0, places=3)
        self.assertAlmostEqual(gainsR[W], 2/5, places=3)

    def test_id3_depth_2(self):
        # max depth of 2 coincidentally produces the same tree as max depth 3
        # max depth 3 and higher should always produce the same tree (complete)
        for max_depth in range(2, 5):
            with self.subTest(max_depth=max_depth):
                tree = ID3(majority_error, max_depth, examples, weights, attributes, label)

                assert isinstance(tree, TreeNode)

                self.assertIn(tree.attribute_name, (O.name, H.name))

                if tree.attribute_name == O.name:
                    childS = tree.children["S"]
                    childO = tree.children["O"]
                    childR = tree.children["R"]
                    assert isinstance(childS, TreeNode)
                    assert isinstance(childO, LeafNode)
                    assert isinstance(childR, TreeNode)

                    self.assertEqual(childS.attribute_name, H.name)
                    childSH = childS.children["H"]
                    childSN = childS.children["N"]
                    childSL = childS.children["L"]
                    assert isinstance(childSH, LeafNode)
                    assert isinstance(childSN, LeafNode)
                    assert isinstance(childSL, LeafNode)
                    self.assertEqual(childSH.label, "-")
                    self.assertEqual(childSN.label, "+")
                    self.assertEqual(childSL.label, "-")

                    self.assertEqual(childO.label, "+")

                    self.assertEqual(childR.attribute_name, W.name)
                    childRS = childR.children["S"]
                    childRW = childR.children["W"]
                    assert isinstance(childRS, LeafNode)
                    assert isinstance(childRW, LeafNode)
                    self.assertEqual(childRS.label, "-")
                    self.assertEqual(childRW.label, "+")
                elif tree.attribute_name == H.name:
                    self.skipTest("Have not reasoned through initial H split.")

    def test_id3_depth_3_prediction(self):
        tree = ID3(majority_error, 3, examples, weights, attributes, label)
        assert isinstance(tree, TreeNode)
        self.assertIn(tree.attribute_name, (O.name, H.name))

        if tree.attribute_name == O.name:
            predictor = lambda e: predict(tree, e)
            self.assertListEqual(
                list(map(predictor, examples)),
                ["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"]
            )
            self.assertEqual(evaluate(predictor, examples, weights, label), 1)
        elif tree.attribute_name == H.name:
            self.skipTest("Have not reasoned through initial H split.")

if __name__ == "__main__":
    unittest.main()
