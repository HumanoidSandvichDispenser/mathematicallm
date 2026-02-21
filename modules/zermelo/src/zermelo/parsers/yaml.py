"""YAML parser for game trees."""

from collections.abc import Mapping
from pathlib import Path

import yaml
from sympy import Rational

from zermelo.trees.node import (
    Node,
    DecisionNode,
    ChanceNode,
    TerminalNode,
    InformationSet,
)


def load_game_from_yaml(source: str | Path) -> Node:
    """
    Load a game tree from a YAML file or string.

    YAML format:
        # Root node (required)
        root:
            type: decision  # decision, chance, terminal
            label: root     # node label
            player: 0       # for decision nodes
            information_set: shared  # optional, for decision nodes
            payoffs: [1, 2]  # for terminal nodes
            probabilities:  # for chance nodes
                left: 0.5
                right: 0.5
            children:
                left:
                    type: terminal
                    label: left
                    payoffs: [1, 2]
                right:
                    ...

        # Information sets (optional, for grouping decision nodes)
        information_sets:
            shared:
                player: 1
                nodes: [node_a, node_b]
    """
    if isinstance(source, Path):
        with open(source) as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(source)

    if data is None:
        raise ValueError("Empty YAML file")

    if "root" not in data:
        raise ValueError("YAML must contain a 'root' key")

    info_sets: dict[str, InformationSet] = {}
    if "information_sets" in data:
        for label, spec in data["information_sets"].items():
            info_sets[label] = InformationSet(label, spec["player"])

    node_registry: dict[str, Node] = {}
    pending_info_set_nodes: list[tuple[DecisionNode, str]] = []

    def parse_node(spec: dict, parent_info_set_label: str | None = None) -> Node:
        node_type = spec.get("type", "decision")
        label = spec.get("label")
        info_set_label = spec.get("information_set")

        if node_type == "terminal":
            payoffs = _parse_payoffs(spec.get("payoffs", []))
            node = TerminalNode(label, payoffs)
            node_registry[label] = node
            return node

        if node_type == "chance":
            prob_map = _parse_probabilities(spec.get("probabilities", {}))
            node = ChanceNode(label, prob_map)
            node_registry[label] = node
            _add_children(node, spec.get("children", {}))
            return node

        if node_type == "decision":
            player = spec.get("player")
            info_set_label = spec.get("information_set")
            node = DecisionNode(label, player)
            node_registry[label] = node
            pending_info_set_nodes.append((node, info_set_label))
            _add_children(node, spec.get("children", {}))
            return node

        raise ValueError(f"Unknown node type: {node_type}")

    def _add_children(parent: Node, children_spec: dict):
        for action, child_spec in children_spec.items():
            child = parse_node(child_spec)
            parent.add_child(child, action)

    root_spec = data["root"]
    root = parse_node(root_spec)

    for node, info_set_label in pending_info_set_nodes:
        if info_set_label and info_set_label in info_sets:
            node.information_set = info_sets[info_set_label]
            info_sets[info_set_label].add_node(node)

    return root


def _parse_payoffs(payoffs: list) -> tuple:
    result = []
    for p in payoffs:
        if isinstance(p, int):
            result.append(p)
        elif isinstance(p, float):
            result.append(Rational(p))
        elif isinstance(p, str):
            result.append(Rational(p))
        else:
            result.append(p)
    return tuple(result)


def _parse_probabilities(probs: dict) -> dict[str, Rational]:
    result = {}
    for action, prob in probs.items():
        if isinstance(prob, (int, float)):
            result[action] = Rational(prob)
        elif isinstance(prob, str):
            result[action] = Rational(prob)
        else:
            result[action] = prob
    return result
