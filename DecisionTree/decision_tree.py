#!/usr/bin/env python3

from typing import Dict, Union
from dataclasses import dataclass, field

from dataset.dataset import Example, Examples, Attribute, AttributeName, AttributeValue

@dataclass
class TreeNode:
    attribute_name: AttributeName
    children: Dict[AttributeValue, 'Node'] = field(default_factory=dict)

    def add_child(self, attribute_value: AttributeValue, child: 'Node'):
        self.children[attribute_value] = child

@dataclass
class LeafNode:
    label: AttributeValue

Node = Union[TreeNode, LeafNode]

def predict(tree: Node, example: Example) -> AttributeValue:
    if isinstance(tree, LeafNode):
        return tree.label
    else:
        attribute_value = example[tree.attribute_name]
        return predict(tree.children[attribute_value], example)

def height(tree: Node) -> int:
    if isinstance(tree, LeafNode):
        return 1
    else:
        return max(map(height, tree.children.values())) + 1
