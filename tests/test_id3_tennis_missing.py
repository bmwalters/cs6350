#!/usr/bin/env python3

import unittest

from dataset.dataset import Attribute, partition, evaluate
from DecisionTree.decision_tree import LeafNode, TreeNode, predict
from DecisionTree.id3 import ID3, attribute_gains, entropy

O = Attribute("O", set(("S", "O", "R")))
T = Attribute("T", set(("H", "M", "C")))
H = Attribute("H", set(("H", "N", "L")))
W = Attribute("W", set(("S", "W")))
attributes = set((O, T, H, W))
label = Attribute("Play?", set(("-", "+")))

raw_examples = [
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
    { "O": "_", "T": "M", "H": "N", "W": "W", "Play?": "+" },
]
raw_weights = [1.0 for _ in raw_examples]

class TestTennisClassifier(unittest.TestCase):
    def test_entropy_and_gain_picking_most_common_globally(self):
        new_examples = [e.copy() for e in raw_examples]
        new_examples[-1]["O"] = "S"

        result = entropy(new_examples, raw_weights, label.name)
        self.assertAlmostEqual(result, 0.918, places=3)

        gains = attribute_gains(entropy, new_examples, raw_weights, attributes, label)
        self.assertAlmostEqual(gains[O], 0.19, places=2)
        self.assertAlmostEqual(gains[T], 0.03, places=2)
        self.assertAlmostEqual(gains[H], 0.17, places=2)
        self.assertAlmostEqual(gains[W], 0.06, places=2)

    def test_entropy_and_gain_picking_most_common_among_label(self):
        new_examples = [e.copy() for e in raw_examples]
        new_examples[-1]["O"] = "O"

        result = entropy(new_examples, raw_weights, label.name)
        self.assertAlmostEqual(result, 0.918, places=3)

        gains = attribute_gains(entropy, new_examples, raw_weights, attributes, label)
        self.assertAlmostEqual(gains[O], 0.27, places=2)
        self.assertAlmostEqual(gains[T], 0.03, places=2)
        self.assertAlmostEqual(gains[H], 0.17, places=2)
        self.assertAlmostEqual(gains[W], 0.06, places=2)

    def build_fractional_examples_and_weights(self):
        fractional_examples = [e.copy() for e in raw_examples]
        fractional_weights = raw_weights.copy()
        example = fractional_examples.pop()
        oweight = fractional_weights.pop()

        exampleS = example.copy()
        exampleS["O"] = "S"
        weightS = 5/14

        exampleO = example.copy()
        exampleO["O"] = "O"
        weightO = 4/14

        exampleR = example.copy()
        exampleR["O"] = "R"
        weightR = 5/14

        self.assertAlmostEqual(weightS + weightO + weightR, oweight)

        fractional_examples += [exampleS, exampleO, exampleR]
        fractional_weights += [weightS, weightO, weightR]

        return fractional_examples, fractional_weights

    def test_entropy_and_gain_fractional_counts(self):
        fractional_examples, fractional_weights = self.build_fractional_examples_and_weights()

        result = entropy(fractional_examples, fractional_weights, label.name)
        self.assertAlmostEqual(result, 0.918, places=3)

        gains = attribute_gains(entropy, fractional_examples, fractional_weights, attributes, label)
        self.assertAlmostEqual(gains[O], 0.224, places=3)
        self.assertAlmostEqual(gains[T], 0.03, places=2)
        self.assertAlmostEqual(gains[H], 0.17, places=2)
        self.assertAlmostEqual(gains[W], 0.06, places=2)

    def test_id3_depth_1_fractional_counts(self):
        fractional_examples, fractional_weights = self.build_fractional_examples_and_weights()

        tree = ID3(entropy, 1, fractional_examples, fractional_weights, attributes, label)

        assert isinstance(tree, TreeNode)

        self.assertEqual(tree.attribute_name, O.name)

        childS = tree.children["S"]
        childO = tree.children["O"]
        childR = tree.children["R"]
        assert isinstance(childS, LeafNode)
        assert isinstance(childO, LeafNode)
        assert isinstance(childR, LeafNode)
        self.assertEqual(childS.label, "-")
        self.assertEqual(childO.label, "+")
        self.assertEqual(childR.label, "+")

    def test_entropy_gains_partitioned_O_fractional_counts(self):
        fractional_examples, fractional_weights = self.build_fractional_examples_and_weights()

        SvS, WvS = partition(fractional_examples, fractional_weights, O.name, "S")
        SvO, WvO = partition(fractional_examples, fractional_weights, O.name, "O")
        SvR, WvR = partition(fractional_examples, fractional_weights, O.name, "R")

        self.assertEqual(SvS, [
            fractional_examples[0],
            fractional_examples[1],
            fractional_examples[7],
            fractional_examples[8],
            fractional_examples[10],
            fractional_examples[14],
        ])
        self.assertEqual(SvO, [
            fractional_examples[2],
            fractional_examples[6],
            fractional_examples[11],
            fractional_examples[12],
            fractional_examples[15],
        ])
        self.assertEqual(SvR, [
            fractional_examples[3],
            fractional_examples[4],
            fractional_examples[5],
            fractional_examples[9],
            fractional_examples[13],
            fractional_examples[16],
        ])

        self.assertEqual(WvS, [1, 1, 1, 1, 1, 5/14])
        self.assertEqual(WvO, [1, 1, 1, 1, 4/14])
        self.assertEqual(WvR, [1, 1, 1, 1, 1, 5/14])

        self.assertAlmostEqual(entropy(SvS, WvS, label.name), 0.990, places=3)
        self.assertAlmostEqual(entropy(SvO, WvO, label.name), 0, places=3)
        self.assertAlmostEqual(entropy(SvR, WvR, label.name), 0.953, places=3)

        gainsS = attribute_gains(entropy, SvS, WvS, set((T, H, W)), label)
        self.assertAlmostEqual(gainsS[T], 0.557, places=3)
        self.assertAlmostEqual(gainsS[H], 0.990, places=3)
        self.assertGreater(gainsS[H], gainsS[W])

        gainsR = attribute_gains(entropy, SvR, WvR, set((T, H, W)), label)
        self.assertAlmostEqual(gainsR[T], 0.029, places=3)
        self.assertAlmostEqual(gainsR[H], 0.029, places=3)
        self.assertAlmostEqual(gainsR[W], 0.953, places=3)

    # this generates the same tree as _gi and _me without missing.
    def test_id3_depth_2_fractional_counts(self):
        fractional_examples, fractional_weights = self.build_fractional_examples_and_weights()

        # max depth of 2 coincidentally produces the same tree as max depth 3
        # max depth 3 and higher should always produce the same tree (complete)
        for max_depth in range(2, 5):
            with self.subTest(max_depth=max_depth):
                tree = ID3(entropy, max_depth, fractional_examples, fractional_weights, attributes, label)

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

if __name__ == "__main__":
    unittest.main()
