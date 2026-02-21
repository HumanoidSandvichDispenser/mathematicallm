"""
Tests for strategies.py - finding full and reduced pure strategies.
"""

import pytest
from sympy import S
from zermelo.trees.node import DecisionNode, ChanceNode, TerminalNode
from zermelo.trees.strategy import Strategy
from zermelo.analysis.strategies import (
    find_full_pure_strategies,
    find_reduced_pure_strategies,
    create_payoff_array,
)


class TestFindFullPureStrategies:
    def test_terminal_node(self):
        """Terminal node returns empty strategy."""
        node = TerminalNode("end", (S(1), S(2)))
        result = find_full_pure_strategies(node, player=0)
        assert result == [Strategy({})]

    def test_single_decision_node_player_owns(self):
        """Player owns single decision node - returns strategies for each action."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "left")
        root.add_child(right, "right")

        result = find_full_pure_strategies(root, player=0)
        assert len(result) == 2
        assert Strategy({"root": "left"}) in result
        assert Strategy({"root": "right"}) in result

    def test_single_decision_node_player_does_not_own(self):
        """Player doesn't own decision node - returns single empty strategy."""
        root = DecisionNode("root", player=1)
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "left")
        root.add_child(right, "right")

        result = find_full_pure_strategies(root, player=0)
        assert result == [Strategy({})]

    def test_two_decision_nodes_same_player(self):
        """Two decision nodes owned by same player - returns product."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = DecisionNode("n1", player=0)
        a = TerminalNode("a", (S(3), S(4)))
        b = TerminalNode("b", (S(5), S(6)))
        right.add_child(a, "a")
        right.add_child(b, "b")
        root.add_child(left, "left")
        root.add_child(right, "right")

        result = find_full_pure_strategies(root, player=0)
        assert len(result) == 4
        assert Strategy({"n1": "a", "root": "left"}) in result
        assert Strategy({"n1": "b", "root": "left"}) in result
        assert Strategy({"n1": "a", "root": "right"}) in result
        assert Strategy({"n1": "b", "root": "right"}) in result

    def test_two_decision_nodes_different_players(self):
        """Two decision nodes owned by different players."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = DecisionNode("n1", player=1)
        a = TerminalNode("a", (S(3), S(4)))
        b = TerminalNode("b", (S(5), S(6)))
        right.add_child(a, "a")
        right.add_child(b, "b")
        root.add_child(left, "left")
        root.add_child(right, "right")

        player_0_strats = find_full_pure_strategies(root, player=0)
        assert len(player_0_strats) == 2
        assert Strategy({"root": "left"}) in player_0_strats
        assert Strategy({"root": "right"}) in player_0_strats

        player_1_strats = find_full_pure_strategies(root, player=1)
        assert len(player_1_strats) == 2
        assert Strategy({"n1": "a"}) in player_1_strats
        assert Strategy({"n1": "b"}) in player_1_strats

    def test_chance_node(self):
        """Chance node - player gets empty strategy."""
        chance = ChanceNode("chance", {"left": S(1) / 2, "right": S(1) / 2})
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        chance.add_child(left, "left")
        chance.add_child(right, "right")

        result = find_full_pure_strategies(chance, player=0)
        assert result == [Strategy({})]

    def test_mixed_chance_and_decision(self):
        """Chance node followed by decision node."""
        root = ChanceNode("chance", {"L": S(1) / 2, "R": S(1) / 2})
        left = DecisionNode("n1", player=0)
        right = TerminalNode("right", (S(3), S(4)))
        a = TerminalNode("a", (S(1), S(2)))
        b = TerminalNode("b", (S(5), S(6)))
        left.add_child(a, "a")
        left.add_child(b, "b")
        root.add_child(left, "L")
        root.add_child(right, "R")

        result = find_full_pure_strategies(root, player=0)
        assert len(result) == 2
        assert Strategy({"n1": "a"}) in result
        assert Strategy({"n1": "b"}) in result

    def test_deeply_nested_tree(self):
        """Three levels of nesting."""
        #       root (P0)
        #       /    \
        #    left    right (P0)
        #    / \     / \
        #   t1  t2  t3  t4
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = DecisionNode("n1", player=0)
        t3 = TerminalNode("t3", (S(3), S(4)))
        t4 = TerminalNode("t4", (S(5), S(6)))
        right.add_child(t3, "t3")
        right.add_child(t4, "t4")
        root.add_child(left, "left")
        root.add_child(right, "right")

        result = find_full_pure_strategies(root, player=0)
        assert len(result) == 4
        assert Strategy({"n1": "t3", "root": "left"}) in result
        assert Strategy({"n1": "t4", "root": "left"}) in result
        assert Strategy({"n1": "t3", "root": "right"}) in result
        assert Strategy({"n1": "t4", "root": "right"}) in result

    def test_three_players(self):
        """Three players in sequence."""
        #       P0
        #       / \
        #    P1   terminal
        #    / \
        # P2  terminal
        # / \
        # t1 t2
        root = DecisionNode("root", player=0)
        left = DecisionNode("n1", player=1)
        right = TerminalNode("right", (S(1), S(2), S(3)))
        p2 = DecisionNode("n2", player=2)
        t1 = TerminalNode("t1", (S(4), S(5), S(6)))
        t2 = TerminalNode("t2", (S(7), S(8), S(9)))
        p2.add_child(t1, "t1")
        p2.add_child(t2, "t2")
        left.add_child(p2, "p2")
        left.add_child(TerminalNode("t3", (S(10), S(11), S(12))), "t3")
        root.add_child(left, "left")
        root.add_child(right, "right")

        p0_strats = find_full_pure_strategies(root, player=0)
        assert len(p0_strats) == 2

        p1_strats = find_full_pure_strategies(root, player=1)
        assert len(p1_strats) == 2

        p2_strats = find_full_pure_strategies(root, player=2)
        assert len(p2_strats) == 2

    def test_no_decision_nodes_for_player(self):
        """Tree has no decision nodes for this player."""
        root = DecisionNode("root", player=1)
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "left")
        root.add_child(right, "right")

        result = find_full_pure_strategies(root, player=0)
        assert result == [Strategy({})]

    def test_shared_information_set(self):
        """Two decision nodes in same information set."""
        from zermelo.trees.node import InformationSet

        info_set = InformationSet("shared", player=0)
        root = DecisionNode("root", player=0, information_set=info_set)
        other = DecisionNode("other", player=0, information_set=info_set)
        t1 = TerminalNode("t1", (S(1), S(2)))
        t2 = TerminalNode("t2", (S(3), S(4)))
        t3 = TerminalNode("t3", (S(5), S(6)))
        t4 = TerminalNode("t4", (S(7), S(8)))
        root.add_child(t1, "a")
        root.add_child(t2, "b")
        other.add_child(t3, "a")
        other.add_child(t4, "b")

        result = find_full_pure_strategies(root, player=0)
        assert len(result) == 2
        assert Strategy({"shared": "a"}) in result
        assert Strategy({"shared": "b"}) in result


class TestFindReducedPureStrategies:
    def test_terminal_node(self):
        """Terminal node returns empty strategy."""
        node = TerminalNode("end", (S(1), S(2)))
        result = find_reduced_pure_strategies(node, player=0)
        assert result == [Strategy({})]

    def test_single_decision_node(self):
        """Single decision node - same as full pure."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "left")
        root.add_child(right, "right")

        result = find_reduced_pure_strategies(root, player=0)
        assert len(result) == 2
        assert Strategy({"root": "left"}) in result
        assert Strategy({"root": "right"}) in result

    def test_action_leads_to_more_decisions(self):
        """When action leads to subtree with more decisions."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = DecisionNode("n1", player=0)
        a = TerminalNode("a", (S(3), S(4)))
        b = TerminalNode("b", (S(5), S(6)))
        right.add_child(a, "a")
        right.add_child(b, "b")
        root.add_child(left, "left")
        root.add_child(right, "right")

        full = find_full_pure_strategies(root, player=0)
        reduced = find_reduced_pure_strategies(root, player=0)

        assert len(full) == 4
        assert len(reduced) == 3

        assert Strategy({"root": "left"}) in reduced
        assert Strategy({"n1": "a", "root": "right"}) in reduced
        assert Strategy({"n1": "b", "root": "right"}) in reduced

    def test_different_players(self):
        """Reduced strategies for different players."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = DecisionNode("n1", player=1)
        a = TerminalNode("a", (S(3), S(4)))
        b = TerminalNode("b", (S(5), S(6)))
        right.add_child(a, "a")
        right.add_child(b, "b")
        root.add_child(left, "left")
        root.add_child(right, "right")

        p0_reduced = find_reduced_pure_strategies(root, player=0)
        p1_reduced = find_reduced_pure_strategies(root, player=1)

        assert len(p0_reduced) == 2
        assert len(p1_reduced) == 2

    def test_chance_node(self):
        """Chance node - player gets empty strategy."""
        chance = ChanceNode("chance", {"left": S(1) / 2, "right": S(1) / 2})
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        chance.add_child(left, "left")
        chance.add_child(right, "right")

        result = find_reduced_pure_strategies(chance, player=0)
        assert result == [Strategy({})]

    def test_mixed_chance_and_decision(self):
        """Chance node followed by decision node."""
        root = ChanceNode("chance", {"L": S(1) / 2, "R": S(1) / 2})
        left = DecisionNode("n1", player=0)
        right = TerminalNode("right", (S(3), S(4)))
        a = TerminalNode("a", (S(1), S(2)))
        b = TerminalNode("b", (S(5), S(6)))
        left.add_child(a, "a")
        left.add_child(b, "b")
        root.add_child(left, "L")
        root.add_child(right, "R")

        result = find_reduced_pure_strategies(root, player=0)
        assert len(result) == 2
        assert Strategy({"n1": "a"}) in result
        assert Strategy({"n1": "b"}) in result

    def test_deeply_nested(self):
        """Three levels of nesting."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = DecisionNode("n1", player=0)
        t3 = TerminalNode("t3", (S(3), S(4)))
        t4 = TerminalNode("t4", (S(5), S(6)))
        right.add_child(t3, "t3")
        right.add_child(t4, "t4")
        root.add_child(left, "left")
        root.add_child(right, "right")

        full = find_full_pure_strategies(root, player=0)
        reduced = find_reduced_pure_strategies(root, player=0)

        assert len(full) == 4
        assert len(reduced) == 3

    def test_no_decision_nodes_for_player(self):
        """Tree has no decision nodes for this player."""
        root = DecisionNode("root", player=1)
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "left")
        root.add_child(right, "right")

        result = find_reduced_pure_strategies(root, player=0)
        assert result == [Strategy({})]

    def test_player_multiple_info_sets_different_branches(self):
        """Player has multiple info sets in different branches - strategies should be combined.

        This is the bug case: when player 1 has info sets 'o' and 'm' in different
        branches, the reduced strategies need to include BOTH info sets in each strategy.
        """
        root = DecisionNode("root", player=0)
        o_node = DecisionNode("o", player=1)
        m_node = DecisionNode("m", player=1)
        o_o = TerminalNode("oo", (S(2), S(1)))
        o_m = TerminalNode("om", (S(0), S(0)))
        m_o = TerminalNode("mo", (S(0), S(0)))
        m_m = TerminalNode("mm", (S(1), S(2)))
        o_node.add_child(o_o, "O")
        o_node.add_child(o_m, "M")
        m_node.add_child(m_o, "O")
        m_node.add_child(m_m, "M")
        root.add_child(o_node, "O")
        root.add_child(m_node, "M")

        p1_reduced = find_reduced_pure_strategies(root, player=1)

        assert len(p1_reduced) == 4
        for s in p1_reduced:
            assert "o" in s
            assert "m" in s

    def test_payoff_array_with_reduced_strategies_multiple_info_sets(self):
        """Create payoff array with reduced strategies that have multiple info sets."""
        root = DecisionNode("root", player=0)
        o_node = DecisionNode("o", player=1)
        m_node = DecisionNode("m", player=1)
        o_o = TerminalNode("oo", (S(2), S(1)))
        o_m = TerminalNode("om", (S(0), S(0)))
        m_o = TerminalNode("mo", (S(0), S(0)))
        m_m = TerminalNode("mm", (S(1), S(2)))
        o_node.add_child(o_o, "O")
        o_node.add_child(o_m, "M")
        m_node.add_child(m_o, "O")
        m_node.add_child(m_m, "M")
        root.add_child(o_node, "O")
        root.add_child(m_node, "M")

        p0_reduced = find_reduced_pure_strategies(root, player=0)
        p1_reduced = find_reduced_pure_strategies(root, player=1)

        array = create_payoff_array(root, [p0_reduced, p1_reduced])

        assert array.shape == (2, 4, 2)


class TestCreatePayoffArray:
    def test_two_player_game(self):
        """Simple two-player game with 2 strategies each."""
        root = DecisionNode("root", player=0)
        left = DecisionNode("left", player=1)
        right = DecisionNode("right", player=1)
        root.add_child(left, "L")
        root.add_child(right, "R")
        left.add_child(TerminalNode("t1", (S(1), S(2))), "U")
        left.add_child(TerminalNode("t2", (S(3), S(4))), "D")
        right.add_child(TerminalNode("t3", (S(5), S(6))), "U")
        right.add_child(TerminalNode("t4", (S(7), S(8))), "D")

        p0_strats = find_full_pure_strategies(root, player=0)
        p1_strats = find_full_pure_strategies(root, player=1)

        array = create_payoff_array(root, [p0_strats, p1_strats])

        assert array.shape == (2, 4, 2)
        assert tuple(array[0, 0]) == (S(1), S(2))
        assert tuple(array[0, 1]) == (S(1), S(2))
        assert tuple(array[0, 2]) == (S(3), S(4))
        assert tuple(array[0, 3]) == (S(3), S(4))
        assert tuple(array[1, 0]) == (S(5), S(6))
        assert tuple(array[1, 1]) == (S(7), S(8))
        assert tuple(array[1, 2]) == (S(5), S(6))
        assert tuple(array[1, 3]) == (S(7), S(8))

    def test_single_player(self):
        """Single player game."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1),))
        right = TerminalNode("right", (S(3),))
        root.add_child(left, "L")
        root.add_child(right, "R")

        p0_strats = find_full_pure_strategies(root, player=0)

        array = create_payoff_array(root, [p0_strats])

        assert array.shape == (2, 1)
        assert tuple(array[0]) == (S(1),)
        assert tuple(array[1]) == (S(3),)

    def test_three_player_game(self):
        """Three-player game."""
        #    P0
        #   /  \
        #  P1   t1
        #  / \
        # t2 t3
        root = DecisionNode("root", player=0)
        left = DecisionNode("n1", player=1)
        right = TerminalNode("t1", (S(1), S(2), S(3)))
        root.add_child(left, "L")
        root.add_child(right, "R")
        left.add_child(TerminalNode("t2", (S(4), S(5), S(6))), "U")
        left.add_child(TerminalNode("t3", (S(7), S(8), S(9))), "D")

        p0_strats = find_full_pure_strategies(root, player=0)
        p1_strats = find_full_pure_strategies(root, player=1)

        array = create_payoff_array(root, [p0_strats, p1_strats])

        assert array.shape == (2, 2, 3)
        assert tuple(array[0, 0]) == (S(4), S(5), S(6))
        assert tuple(array[0, 1]) == (S(7), S(8), S(9))
        assert tuple(array[1, 0]) == (S(1), S(2), S(3))
        assert tuple(array[1, 1]) == (S(1), S(2), S(3))

    def test_chance_node(self):
        """Game with chance node."""
        root = ChanceNode("chance", {"L": S(1) / 2, "R": S(1) / 2})
        left = TerminalNode("left", (S(1), S(2)))
        right = TerminalNode("right", (S(3), S(4)))
        root.add_child(left, "L")
        root.add_child(right, "R")

        p0_strats = find_full_pure_strategies(root, player=0)

        array = create_payoff_array(root, [p0_strats])

        assert array.shape == (1, 2)
        assert tuple(array[0]) == (S(2), S(3))

    def test_with_reduced_strategies(self):
        """Using reduced strategies."""
        root = DecisionNode("root", player=0)
        left = TerminalNode("left", (S(1), S(2)))
        right = DecisionNode("n1", player=0)
        right.add_child(TerminalNode("a", (S(3), S(4))), "a")
        right.add_child(TerminalNode("b", (S(5), S(6))), "b")
        root.add_child(left, "L")
        root.add_child(right, "R")

        p0_full = find_full_pure_strategies(root, player=0)
        p0_reduced = find_reduced_pure_strategies(root, player=0)

        full_array = create_payoff_array(root, [p0_full])
        reduced_array = create_payoff_array(root, [p0_reduced])

        assert full_array.shape == (4, 2)
        assert reduced_array.shape == (3, 2)
