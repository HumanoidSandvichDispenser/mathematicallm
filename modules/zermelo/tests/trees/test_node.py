"""
Tests for node.py - tree node classes.
"""

import pytest
import sympy
from sympy import S
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

    def apply_strategy(self, strategies: list[Strategy]) -> tuple:
        raise NotImplementedError("NodeForTests is for testing only")

    def _deep_copy(
        self, info_set_map: dict[InformationSet, InformationSet]
    ) -> "NodeForTests":
        new_node = NodeForTests(self.label)
        for action, child in self.children.items():
            new_node.children[action] = child._deep_copy(info_set_map)
        return new_node


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
        assert isinstance(payoff, tuple)
        assert payoff == (S(1), S(2))

    def test_apply_strategy_other_action(self):
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "left")
        root.add_child(right, "right")

        strategies = [Strategy({"root": "right"})]
        payoff = root.apply_strategy(strategies)
        assert isinstance(payoff, tuple)
        assert payoff == (S(3), S(4))

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

    def test_is_perfect_information_single_node_info_set(self):
        node = DecisionNode("root", player="p0")
        t1 = TerminalNode("t1", (S(1),))
        t2 = TerminalNode("t2", (S(2),))
        node.add_child(t1, "left")
        node.add_child(t2, "right")

        assert node.is_perfect_information is True

    def test_is_perfect_information_multi_node_info_set(self):
        info_set = InformationSet("shared", player="p0")
        node1 = DecisionNode("n1", player="p0", information_set=info_set)
        node2 = DecisionNode("n2", player="p0", information_set=info_set)

        t1 = TerminalNode("t1", (S(1),))
        t2 = TerminalNode("t2", (S(2),))
        node1.add_child(t1, "left")
        node1.add_child(t2, "right")
        node2.add_child(t1, "left")
        node2.add_child(t2, "right")

        assert node1.is_perfect_information is False
        assert node2.is_perfect_information is False

    def test_is_perfect_information_tree_perfect_info(self):
        root = DecisionNode("root", player="p0")
        left = DecisionNode("left", player="p1")
        right = DecisionNode("right", player="p1")
        root.add_child(left, "left")
        root.add_child(right, "right")
        left.add_child(TerminalNode("t1", (S(1),)), "up")
        left.add_child(TerminalNode("t2", (S(2),)), "down")
        right.add_child(TerminalNode("t3", (S(3),)), "up")
        right.add_child(TerminalNode("t4", (S(4),)), "down")

        assert root.is_perfect_information is True

    def test_is_perfect_information_recursive(self):
        info_set = InformationSet("shared", player="p1")
        root = DecisionNode("root", player="p0")
        left = DecisionNode("left", player="p1", information_set=info_set)
        right = DecisionNode("right", player="p1", information_set=info_set)
        root.add_child(left, "left")
        root.add_child(right, "right")
        left.add_child(TerminalNode("t1", (S(1),)), "up")
        right.add_child(TerminalNode("t2", (S(2),)), "up")

        assert root.is_perfect_information is False


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
        assert isinstance(node.payoffs, tuple)

    def test_init_with_sympy_expressions(self):
        payoffs = (sympy.Rational(1, 2), sympy.Symbol("x"))
        node = TerminalNode("end", payoffs)
        assert isinstance(node.payoffs, tuple)

    def test_init_with_matrix(self):
        node = TerminalNode("end", (S(1), S(2)))
        assert isinstance(node.payoffs, tuple)

    def test_apply_strategy(self):
        node = TerminalNode("end", (S(1), S(2), S(3)))
        strategies = [Strategy({}), Strategy({})]
        payoff = node.apply_strategy(strategies)
        assert isinstance(payoff, tuple)
        assert payoff == (S(1), S(2), S(3))

    def test_is_perfect_information_true(self):
        node = TerminalNode("end", (S(1), S(2)))
        assert node.is_perfect_information is True


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


class TestDeepCopy:
    def test_copy_decision_node(self):
        root = DecisionNode("root", player="p0")
        root.add_child(TerminalNode("t1", (S(1),)), "left")
        root.add_child(TerminalNode("t2", (S(2),)), "right")

        copied = root.copy()

        assert copied is not root
        assert copied.label == "root"
        assert copied.player == "p0"
        assert set(copied.children.keys()) == {"left", "right"}

    def test_copy_decision_node_different_objects(self):
        root = DecisionNode("root", player="p0")
        root.add_child(TerminalNode("t1", (S(1),)), "left")

        copied = root.copy()

        assert root is not copied
        assert root.children["left"] is not copied.children["left"]

    def test_copy_shared_info_set(self):
        info_set = InformationSet("shared", player="p1")
        left = DecisionNode("left", player="p1", information_set=info_set)
        right = DecisionNode("right", player="p1", information_set=info_set)

        root = DecisionNode("root", player="p0")
        root.add_child(left, "left")
        root.add_child(right, "right")

        left.add_child(TerminalNode("t1", (S(1),)), "a")
        right.add_child(TerminalNode("t2", (S(2),)), "a")

        copied = root.copy()

        copied_left = copied.children["left"]
        copied_right = copied.children["right"]

        assert isinstance(copied_left, DecisionNode)
        assert isinstance(copied_right, DecisionNode)

        assert copied_left.information_set is copied_right.information_set
        assert copied_left.information_set.label == "shared"
        assert left.information_set is not copied_left.information_set

    def test_copy_chance_node(self):
        root = DecisionNode("root", player="p0")
        chance = ChanceNode("chance", {"left": S(1) / 2, "right": S(1) / 2})
        root.add_child(chance, "roll")
        chance.add_child(TerminalNode("t1", (S(1),)), "left")
        chance.add_child(TerminalNode("t2", (S(2),)), "right")

        copied = root.copy()

        copied_chance = copied.children["roll"]
        assert isinstance(copied_chance, ChanceNode)
        assert copied_chance is not chance
        assert copied_chance.label == "chance"
        assert copied_chance.probability_map == {"left": S(1) / 2, "right": S(1) / 2}
        assert set(copied_chance.children.keys()) == {"left", "right"}

    def test_copy_chance_node_children_different(self):
        root = DecisionNode("root", player="p0")
        chance = ChanceNode("chance", {"a": S(1)})
        root.add_child(chance, "x")
        chance.add_child(TerminalNode("t1", (S(1),)), "a")

        copied = root.copy()

        assert chance.children["a"] is not copied.children["x"].children["a"]

    def test_copy_terminal_node(self):
        node = TerminalNode("end", (S(1), S(2), S(3)))

        copied = node.copy()

        assert copied is not node
        assert copied.label == "end"
        assert copied.payoffs == (S(1), S(2), S(3))

    def test_copy_full_tree(self):
        root = DecisionNode("root", player="p0")
        left = DecisionNode("left", player="p1")
        right = DecisionNode("right", player="p1")
        root.add_child(left, "left")
        root.add_child(right, "right")

        left.add_child(TerminalNode("t1", (S(1), S(2))), "up")
        left.add_child(TerminalNode("t2", (S(3), S(4))), "down")
        right.add_child(TerminalNode("t3", (S(5), S(6))), "up")
        right.add_child(TerminalNode("t4", (S(7), S(8))), "down")

        copied = root.copy()

        assert copied.label == "root"
        assert copied.player == "p0"
        assert set(copied.children.keys()) == {"left", "right"}

        assert copied.children["left"].label == "left"
        assert copied.children["right"].label == "right"

        assert copied.children["left"].children["up"].payoffs == (S(1), S(2))
        assert copied.children["left"].children["down"].payoffs == (S(3), S(4))
        assert copied.children["right"].children["up"].payoffs == (S(5), S(6))
        assert copied.children["right"].children["down"].payoffs == (S(7), S(8))

    def test_copy_preserves_structure_not_references(self):
        root = DecisionNode("root", player="p0")
        t1 = TerminalNode("t1", (S(1),))
        t2 = TerminalNode("t2", (S(2),))
        root.add_child(t1, "left")
        root.add_child(t2, "right")

        copied = root.copy()

        assert root.children["left"] is not copied.children["left"]
        assert root.children["right"] is not copied.children["right"]
        assert t1 is not copied.children["left"]
        assert t2 is not copied.children["right"]

    def test_copy_partial_tree_from_shared_info_set(self):
        info_set = InformationSet("shared", player="p1")
        node1 = DecisionNode("node1", player="p1", information_set=info_set)
        node2 = DecisionNode("node2", player="p1", information_set=info_set)

        node1.add_child(TerminalNode("t1", (S(1),)), "a")
        node2.add_child(TerminalNode("t2", (S(2),)), "a")

        assert len(info_set.nodes) == 2

        copied = node1.copy()

        assert len(info_set.nodes) == 2
        assert copied.information_set is not info_set
        assert copied.information_set.label == "shared"
        assert len(copied.information_set.nodes) == 1
        assert copied in copied.information_set.nodes
