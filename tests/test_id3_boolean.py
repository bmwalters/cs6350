#!/usr/bin/env python3

import unittest

from dataset.dataset import Attribute, partition, evaluate
from DecisionTree.decision_tree import LeafNode, TreeNode, predict
from DecisionTree.id3 import ID3, attribute_gains, entropy

x1 = Attribute("x1", set(("0", "1")))
x2 = Attribute("x2", set(("0", "1")))
x3 = Attribute("x3", set(("0", "1")))
x4 = Attribute("x4", set(("0", "1")))
attributes = set((x1, x2, x3, x4))
label = Attribute("y", set(("0", "1")))
examples = [
    { "x1": "0", "x2": "0", "x3": "1", "x4": "0", "y": "0" },
    { "x1": "0", "x2": "1", "x3": "0", "x4": "0", "y": "0" },
    { "x1": "0", "x2": "0", "x3": "1", "x4": "1", "y": "1" },
    { "x1": "1", "x2": "0", "x3": "0", "x4": "1", "y": "1" },
    { "x1": "0", "x2": "1", "x3": "1", "x4": "0", "y": "0" },
    { "x1": "1", "x2": "1", "x3": "0", "x4": "0", "y": "0" },
    { "x1": "0", "x2": "1", "x3": "0", "x4": "1", "y": "0" },
]
# weight _values_ don't matter since all the weights are equal
weights = [1/len(examples) for _ in examples]

class TestBooleanClassifier(unittest.TestCase):
    def test_entropy(self):
        result = entropy(examples, weights, label.name)
        self.assertAlmostEqual(result, 0.863, places=3)

    def test_information_gain(self):
        gains = attribute_gains(entropy, examples, weights, attributes, label)
        self.assertAlmostEqual(gains[x1], 0.062, places=3)
        self.assertAlmostEqual(gains[x2], 0.470, places=3)
        self.assertAlmostEqual(gains[x3], 0.006, places=3)
        self.assertAlmostEqual(gains[x4], 0.470, places=3)

    def test_id3_depth_1(self):
        tree = ID3(entropy, 1, examples, weights, attributes, label)

        assert isinstance(tree, TreeNode)

        self.assertIn(tree.attribute_name, (x2.name, x4.name))

        if tree.attribute_name == x2.name:
            child0, child1 = tree.children["0"], tree.children["1"]
            assert isinstance(child0, LeafNode)
            assert isinstance(child1, LeafNode)
            self.assertEqual(child0.label, "1")
            self.assertEqual(child1.label, "0")
        elif tree.attribute_name == x4.name:
            child0, child1 = tree.children["0"], tree.children["1"]
            assert isinstance(child0, LeafNode)
            assert isinstance(child1, LeafNode)
            self.assertEqual(child0.label, "0")
            self.assertEqual(child1.label, "1")

    def test_entropy_partitioned_x4_1(self):
        Sv, Wv = partition(examples, weights, x4.name, "1")
        self.assertEqual(Sv, [examples[2], examples[3], examples[6]])
        for _, w in zip(range(3), Wv):
            self.assertAlmostEqual(w, 1/len(examples))

        result = entropy(Sv, Wv, label.name)
        self.assertAlmostEqual(result, 0.918, places=3)

    def test_gains_partitioned_x4_1(self):
        Sv, Wv = partition(examples, weights, x4.name, "1")
        gains = attribute_gains(entropy, Sv, Wv, set((x1, x2, x3)), label)
        self.assertAlmostEqual(gains[x1], 0.25, places=2)
        self.assertAlmostEqual(gains[x2], 0.918, places=3)
        self.assertGreater(gains[x2], gains[x3])

    def test_id3_depth_2(self):
        # max depth of 2 or greater should produce the same tree (complete)
        for max_depth in range(2, 5):
            with self.subTest(max_depth=max_depth):
                tree = ID3(entropy, max_depth, examples, weights, attributes, label)

                assert isinstance(tree, TreeNode)

                self.assertIn(tree.attribute_name, (x2.name, x4.name))

                if tree.attribute_name == x2.name:
                    self.skipTest("Have not reasoned through initial x2 split.")
                elif tree.attribute_name == x4.name:
                    child0, child1 = tree.children["0"], tree.children["1"]
                    assert isinstance(child0, LeafNode)
                    assert isinstance(child1, TreeNode)

                    self.assertEqual(child0.label, "0")
                    self.assertEqual(child1.attribute_name, x2.name)

                    child10, child11 = child1.children["0"], child1.children["1"]
                    assert isinstance(child10, LeafNode)
                    assert isinstance(child11, LeafNode)

                    self.assertEqual(child10.label, "1")
                    self.assertEqual(child11.label, "0")

    def test_id3_depth_2_prediction(self):
        tree = ID3(entropy, 2, examples, weights, attributes, label)
        assert isinstance(tree, TreeNode)
        self.assertIn(tree.attribute_name, (x2.name, x4.name))

        if tree.attribute_name == x2.name:
            self.skipTest("Have not reasoned through initial x2 split.")
        elif tree.attribute_name == x4.name:
            predictor = lambda e: predict(tree, e)
            self.assertListEqual(
                list(map(predictor, examples)),
                ["0", "0", "1", "1", "0", "0", "0"]
            )
            self.assertEqual(evaluate(predictor, examples, weights, label), 1)

if __name__ == "__main__":
    unittest.main()
