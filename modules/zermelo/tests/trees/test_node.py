"""
Tests for node.py - tree node classes.
"""

import pytest
import sympy
from sympy import S
from sympy import Matrix
from zermelo.trees.node import (
    Node,
    DecisionNode,
    ChanceNode,
    TerminalNode,
    InformationSet,
)
from zermelo.trees.strategy import Strategy


class NodeForTests(Node):
    def __init__(self, label: str):
        super().__init__(label)

    def apply_strategy(self, strategies: list[Strategy]) -> Matrix:
        raise NotImplementedError("NodeForTests is for testing only")


class TestNodeTests:
    def test_init(self):
        node = NodeForTests("test_label")
        assert node.label == "test_label"
        assert node.parent is None
        assert node.children == {}

    def test_add_child(self):
        parent = NodeForTests("parent")
        child = NodeForTests("child")
        parent.add_child(child, "action1")

        assert child.parent is parent
        assert "action1" in parent.children
        assert parent.children["action1"] is child

    def test_add_child_multiple(self):
        parent = NodeForTests("parent")
        child1 = NodeForTests("child1")
        child2 = NodeForTests("child2")
        parent.add_child(child1, "a1")
        parent.add_child(child2, "a2")

        assert len(parent.children) == 2
        assert parent.children["a1"] is child1
        assert parent.children["a2"] is child2

    def test_rename_action_success(self):
        parent = NodeForTests("parent")
        child = NodeForTests("child")
        parent.add_child(child, "old_action")

        parent.rename_action("old_action", "new_action")

        assert "new_action" in parent.children
        assert "old_action" not in parent.children
        assert parent.children["new_action"] is child

    def test_rename_action_not_found(self):
        parent = NodeForTests("parent")

        with pytest.raises(ValueError, match="Action 'nonexistent' not found"):
            parent.rename_action("nonexistent", "new_action")

    def test_rename_action_already_exists(self):
        parent = NodeForTests("parent")
        child1 = NodeForTests("child1")
        child2 = NodeForTests("child2")
        parent.add_child(child1, "action1")
        parent.add_child(child2, "action2")

        with pytest.raises(ValueError, match="Action 'action2' already exists"):
            parent.rename_action("action1", "action2")

    def test_traverse_postorder_empty(self):
        node = NodeForTests("root")
        result = list(node.traverse_postorder())
        assert result == [node]

    def test_traverse_postorder_single_child(self):
        root = NodeForTests("root")
        child = NodeForTests("child")
        root.add_child(child, "a")

        result = list(root.traverse_postorder())
        assert result == [child, root]

    def test_traverse_postorder_multiple_children(self):
        root = NodeForTests("root")
        child1 = NodeForTests("child1")
        child2 = NodeForTests("child2")
        root.add_child(child1, "a1")
        root.add_child(child2, "a2")

        result = list(root.traverse_postorder())
        assert result == [child1, child2, root]

    def test_traverse_postorder_deep_tree(self):
        root = NodeForTests("root")
        l1 = NodeForTests("l1")
        l2 = NodeForTests("l2")
        l3 = NodeForTests("l3")
        root.add_child(l1, "a1")
        l1.add_child(l2, "b1")
        l2.add_child(l3, "c1")

        result = list(root.traverse_postorder())
        assert result == [l3, l2, l1, root]

    def test_traverse_preorder_empty(self):
        node = NodeForTests("root")
        result = list(node.traverse_preorder())
        assert result == [node]

    def test_traverse_preorder_with_children(self):
        root = NodeForTests("root")
        child = NodeForTests("child")
        root.add_child(child, "a")

        result = list(root.traverse_preorder())
        assert result == [root, child]

    def test_traverse_preorder_deep_tree(self):
        root = NodeForTests("root")
        l1 = NodeForTests("l1")
        l2 = NodeForTests("l2")
        l3 = NodeForTests("l3")
        root.add_child(l1, "a1")
        l1.add_child(l2, "b1")
        l2.add_child(l3, "c1")

        result = list(root.traverse_preorder())
        assert result == [root, l1, l2, l3]


class TestDecisionNode:
    def test_init(self):
        node = DecisionNode("test", player=0)
        assert node.label == "test"
        assert node.player == 0
        assert node.information_set is not None
        assert node.information_set.label == "test"
        assert node.information_set.player == 0
        assert node in node.information_set.nodes

    def test_init_with_custom_information_set(self):
        info_set = InformationSet("shared", player=0)
        node = DecisionNode("test", player=0, information_set=info_set)

        assert node.information_set is info_set
        assert node in info_set.nodes

    def test_actions_no_children(self):
        node = DecisionNode("test", player=0)
        assert node.actions == []

    def test_actions_with_children(self):
        node = DecisionNode("root", player=0)
        node.add_child(DecisionNode("child1", player=1), "left")
        node.add_child(DecisionNode("child2", player=1), "right")

        assert node.actions == ["left", "right"]

    def test_apply_strategy(self):
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "left")
        root.add_child(right, "right")

        strategies = [Strategy({"root": "left"})]
        payoff = root.apply_strategy(strategies)
        assert isinstance(payoff, Matrix)
        assert payoff == Matrix([[S(1)], [S(2)]])

    def test_apply_strategy_other_action(self):
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "left")
        root.add_child(right, "right")

        strategies = [Strategy({"root": "right"})]
        payoff = root.apply_strategy(strategies)
        assert isinstance(payoff, Matrix)
        assert payoff == Matrix([[S(3)], [S(4)]])

    def test_apply_strategy_missing_info_set(self):
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1),))
        root.add_child(left, "left")

        strategies = [Strategy({})]
        with pytest.raises(ValueError, match="does not specify an action"):
            root.apply_strategy(strategies)

    def test_apply_strategy_invalid_action(self):
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1),))
        root.add_child(left, "left")

        strategies = [Strategy({"root": "invalid"})]
        with pytest.raises(ValueError, match="is not available"):
            root.apply_strategy(strategies)


class TestChanceNode:
    def test_init(self):
        probs = {"a": S(1) / 2, "b": S(1) / 2}
        node = ChanceNode("chance", probs)
        assert node.label == "chance"
        assert node.probability_map == probs

    def test_set_probability(self):
        node = ChanceNode("chance", {})
        child = TerminalNode("t", (S(1),))
        node.add_child(child, "action")

        node.set_probability("action", sympy.Rational(1, 3))
        assert node.probability_map["action"] == sympy.Rational(1, 3)

    def test_set_probability_not_found(self):
        node = ChanceNode("chance", {})

        with pytest.raises(ValueError, match="Action 'nonexistent' not found"):
            node.set_probability("nonexistent", sympy.Rational(1, 2))

    def test_add_child_with_probability(self):
        node = ChanceNode("chance", {})
        child = TerminalNode("t", (S(1),))
        node.add_child(child, "action", sympy.Rational(1, 2))

        assert child.parent is node
        assert node.probability_map["action"] == sympy.Rational(1, 2)

    def test_add_child_default_probability(self):
        node = ChanceNode("chance", {})
        child = TerminalNode("t", (S(1),))
        node.add_child(child, "action")

        assert node.probability_map["action"] == sympy.Rational(0)

    def test_rename_action(self):
        node = ChanceNode("chance", {})
        child = TerminalNode("t", (S(1),))
        node.add_child(child, "old", sympy.Rational(1, 2))

        node.rename_action("old", "new")

        assert "new" in node.probability_map
        assert "old" not in node.probability_map
        assert node.probability_map["new"] == sympy.Rational(1, 2)


class TestTerminalNode:
    def test_init(self):
        node = TerminalNode("end", (S(1), S(2), S(3)))
        assert node.label == "end"
        assert isinstance(node.payoffs, Matrix)

    def test_init_with_sympy_expressions(self):
        payoffs = (sympy.Rational(1, 2), sympy.Symbol("x"))
        node = TerminalNode("end", payoffs)
        assert isinstance(node.payoffs, Matrix)

    def test_init_with_vector(self):
        vec = Matrix([S(1), S(2)])
        node = TerminalNode("end", vec)
        assert isinstance(node.payoffs, Matrix)

    def test_apply_strategy(self):
        node = TerminalNode("end", (S(1), S(2), S(3)))
        strategies = [Strategy({}), Strategy({})]
        payoff = node.apply_strategy(strategies)
        assert isinstance(payoff, Matrix)
        assert payoff == Matrix([S(1), S(2), S(3)])


class TestInformationSet:
    def test_init(self):
        info_set = InformationSet("iset", player=0)
        assert info_set.label == "iset"
        assert info_set.player == 0
        assert info_set.nodes == set()

    def test_init_with_nodes(self):
        node1 = DecisionNode("n1", player=0)
        node2 = DecisionNode("n2", player=0)
        info_set = InformationSet("iset", player=0, nodes={node1, node2})

        assert info_set.nodes == {node1, node2}

    def test_add_node(self):
        info_set = InformationSet("iset", player=0)
        node = DecisionNode("n1", player=0, information_set=info_set)

        assert node in info_set.nodes

    def test_add_node_validates_actions(self):
        info_set = InformationSet("iset", player=0)
        node1 = DecisionNode("n1", player=0, information_set=info_set)
        node1.add_child(DecisionNode("c1", player=1), "a1")

        node2 = DecisionNode("n2", player=0)
        node2.add_child(DecisionNode("c2", player=1), "a2")

        with pytest.raises(ValueError, match="same actions"):
            info_set.add_node(node2)

    def test_actions_empty(self):
        info_set = InformationSet("iset", player=0)
        assert info_set.actions == []

    def test_actions_from_nodes(self):
        info_set = InformationSet("iset", player=0)
        node = DecisionNode("n1", player=0, information_set=info_set)
        node.add_child(DecisionNode("c1", player=1), "left")
        node.add_child(DecisionNode("c2", player=1), "right")

        assert info_set.actions == ["left", "right"]

    def test_actions_consistency_across_nodes(self):
        info_set = InformationSet("iset", player=0)
        node1 = DecisionNode("n1", player=0, information_set=info_set)
        node1.add_child(DecisionNode("c1", player=1), "a")

        node2 = DecisionNode("n2", player=0)
        node2.add_child(DecisionNode("c2", player=1), "a")
        info_set.add_node(node2)

        assert info_set.actions == ["a"]
