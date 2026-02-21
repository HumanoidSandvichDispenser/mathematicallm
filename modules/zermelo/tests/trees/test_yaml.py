"""Tests for YAML game tree parser."""

import pytest
from zermelo.trees.node import DecisionNode, ChanceNode, TerminalNode
from zermelo.trees.strategy import Strategy
from sympy import Rational
from zermelo.parsers.yaml import load_game_from_yaml


class TestLoadGameFromYaml:
    def test_simple_decision_tree(self):
        yaml_content = """
root:
    type: decision
    label: root
    player: 0
    children:
        left:
            type: terminal
            label: left
            payoffs: [1, 2]
        right:
            type: terminal
            label: right
            payoffs: [3, 4]
"""
        root = load_game_from_yaml(yaml_content)

        assert isinstance(root, DecisionNode)
        assert root.label == "root"
        assert root.player == 0
        assert "left" in root.children
        assert "right" in root.children
        assert isinstance(root.children["left"], TerminalNode)
        assert root.children["left"].payoffs == (1, 2)
        assert root.children["right"].payoffs == (3, 4)

    def test_chance_node(self):
        yaml_content = """
root:
    type: chance
    label: chance_node
    probabilities:
        left: 0.5
        right: 0.5
    children:
        left:
            type: terminal
            label: left
            payoffs: [1, 2]
        right:
            type: terminal
            label: right
            payoffs: [3, 4]
"""
        root = load_game_from_yaml(yaml_content)

        assert isinstance(root, ChanceNode)
        assert root.label == "chance_node"
        assert root.probability_map["left"] == Rational(1, 2)
        assert root.probability_map["right"] == Rational(1, 2)

    def test_nested_decision_nodes(self):
        yaml_content = """
root:
    type: decision
    label: root
    player: 0
    children:
        left:
            type: terminal
            label: left
            payoffs: [1, 2]
        right:
            type: decision
            label: n1
            player: 1
            children:
                up:
                    type: terminal
                    label: up
                    payoffs: [3, 4]
                down:
                    type: terminal
                    label: down
                    payoffs: [5, 6]
"""
        root = load_game_from_yaml(yaml_content)

        assert isinstance(root, DecisionNode)
        assert isinstance(root.children["right"], DecisionNode)
        assert root.children["right"].player == 1

    def test_information_set(self):
        yaml_content = """
information_sets:
    shared:
        player: 1
        nodes: [node_a, node_b]

root:
    type: decision
    label: root
    player: 0
    children:
        left:
            type: decision
            label: node_a
            player: 1
            information_set: shared
            children:
                up:
                    type: terminal
                    label: up
                    payoffs: [1, 2]
                down:
                    type: terminal
                    label: down
                    payoffs: [3, 4]
        right:
            type: decision
            label: node_b
            player: 1
            information_set: shared
            children:
                up:
                    type: terminal
                    label: up2
                    payoffs: [5, 6]
                down:
                    type: terminal
                    label: down2
                    payoffs: [7, 8]
"""
        root = load_game_from_yaml(yaml_content)

        left = root.children["left"]
        right = root.children["right"]

        assert left.information_set is right.information_set
        assert left.information_set.label == "shared"
        assert left.information_set.player == 1
        assert len(left.information_set.nodes) == 2

    def test_default_information_set(self):
        yaml_content = """
root:
    type: decision
    label: root
    player: 0
    children:
        left:
            type: terminal
            label: left
            payoffs: [1, 2]
        right:
            type: terminal
            label: right
            payoffs: [3, 4]
"""
        root = load_game_from_yaml(yaml_content)

        assert root.information_set.label == "root"
        assert len(root.information_set.nodes) == 1

    def test_rational_payoffs(self):
        yaml_content = """
root:
    type: chance
    label: root
    probabilities:
        left: 1/3
        right: 2/3
    children:
        left:
            type: terminal
            label: left
            payoffs: ["1/2", "3/4"]
        right:
            type: terminal
            label: right
            payoffs: [1, 2]
"""
        root = load_game_from_yaml(yaml_content)

        assert root.probability_map["left"] == Rational(1, 3)
        assert root.children["left"].payoffs == (Rational(1, 2), Rational(3, 4))

    def test_apply_strategy_parsed_tree(self):
        yaml_content = """
root:
    type: decision
    label: root
    player: 0
    children:
        left:
            type: decision
            label: n1
            player: 1
            children:
                up:
                    type: terminal
                    label: up
                    payoffs: [3, 4]
                down:
                    type: terminal
                    label: down
                    payoffs: [5, 6]
        right:
            type: terminal
            label: right
            payoffs: [1, 2]
"""
        root = load_game_from_yaml(yaml_content)

        p0_strat = Strategy({"root": "left"})
        p1_strat = Strategy({"n1": "down"})

        payoff = root.apply_strategy([p0_strat, p1_strat])
        assert payoff == (5, 6)

    def test_missing_root_raises(self):
        yaml_content = """
foo: bar
"""
        with pytest.raises(ValueError, match="must contain a 'root'"):
            load_game_from_yaml(yaml_content)

    def test_empty_yaml_raises(self):
        with pytest.raises(ValueError, match="Empty YAML"):
            load_game_from_yaml("")

    def test_complex_game_with_info_sets(self):
        yaml_content = """
information_sets:
    p1_round1:
        player: 1
        nodes: [r1_p1_c, r1_p1_d]

root:
    type: decision
    label: r1_p0
    player: 0
    children:
        Cooperate:
            type: decision
            label: r1_p1_c
            player: 1
            information_set: p1_round1
            children:
                Cooperate:
                    type: terminal
                    label: t1
                    payoffs: [6, 6]
                Defect:
                    type: terminal
                    label: t2
                    payoffs: [3, 8]
        Defect:
            type: decision
            label: r1_p1_d
            player: 1
            information_set: p1_round1
            children:
                Cooperate:
                    type: terminal
                    label: t3
                    payoffs: [8, 3]
                Defect:
                    type: terminal
                    label: t4
                    payoffs: [1, 1]
"""
        root = load_game_from_yaml(yaml_content)

        assert isinstance(root, DecisionNode)
        assert root.player == 0

        c_child = root.children["Cooperate"]
        d_child = root.children["Defect"]

        assert isinstance(c_child, DecisionNode)
        assert isinstance(d_child, DecisionNode)
        assert c_child.information_set is d_child.information_set
        assert c_child.information_set.label == "p1_round1"

        coop_payoff = root.apply_strategy(
            [Strategy({"r1_p0": "Cooperate"}), Strategy({"p1_round1": "Cooperate"})]
        )
        assert coop_payoff == (6, 6)

        defect_payoff = root.apply_strategy(
            [Strategy({"r1_p0": "Defect"}), Strategy({"p1_round1": "Defect"})]
        )
        assert defect_payoff == (1, 1)
