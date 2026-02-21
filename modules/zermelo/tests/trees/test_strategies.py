"""
Tests for strategies.py - finding full and reduced pure strategies.
"""

import pytest
from sympy import S
from zermelo.trees.node import DecisionNode, ChanceNode, TerminalNode, InformationSet
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

        assert Strategy({"root": "left"}) in p0_reduced
        assert Strategy({"root": "right"}) in p0_reduced
        assert Strategy({"n1": "a"}) in p1_reduced
        assert Strategy({"n1": "b"}) in p1_reduced

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
        """
        Player has multiple singleton information sets in different branches of
        the tree. The strategy should specify an action for each info set, and
        they are independent of each other.
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

        assert Strategy({"o": "O", "m": "O"}) in p1_reduced
        assert Strategy({"o": "O", "m": "M"}) in p1_reduced
        assert Strategy({"o": "M", "m": "O"}) in p1_reduced
        assert Strategy({"o": "M", "m": "M"}) in p1_reduced

    def test_player_same_info_sets_different_branches(self):
        """
        Player has same info set in different branches of the tree. The
        strategy should specify an action for the info set, and it applies to
        both nodes.
        """
        root = DecisionNode("root", player=0)
        info_set = InformationSet("shared", player=1)
        o_node = DecisionNode("o", player=1, information_set=info_set)
        m_node = DecisionNode("m", player=1, information_set=info_set)
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

        assert Strategy({"shared": "O"}) in p1_reduced
        assert Strategy({"shared": "M"}) in p1_reduced

    def test_shared_info_set_siblings_with_subgames(self):
        """
        Regression test: non-player node whose children share a player-owned
        info set, and each child leads to further decisions for that player.

        Tree:
            root (P0)
            ├── A → n_a (P1 @shared)
            │         ├── X → deeper_a (P1 @r2_a)
            │         │         ├── T1
            │         │         └── T2
            │         └── Y → T3
            └── B → n_b (P1 @shared)
                      ├── X → deeper_b (P1 @r2_b)
                      │         ├── T4
                      │         └── T5
                      └── Y → T6

        P1 has info set 'shared' (can't distinguish n_a from n_b).
        If P1 chooses X at 'shared', both deeper_a and deeper_b are reachable
        (from different branches), so P1 needs r2_a and r2_b decisions.
        If P1 chooses Y, neither deeper node is reached, so no r2 decisions needed.

        Correct reduced strategies for P1:
          {shared: Y}                              (Y path omits both r2 info sets)
          {shared: X, r2_a: T1, r2_b: T4}
          {shared: X, r2_a: T1, r2_b: T5}
          {shared: X, r2_a: T2, r2_b: T4}
          {shared: X, r2_a: T2, r2_b: T5}

        The bug caused the non-player node to recurse into both n_a and n_b
        independently and take their Cartesian product, yielding 9 strategies
        instead of 5.
        """
        root = DecisionNode("root", player=0)
        shared = InformationSet("shared", player=1)
        n_a = DecisionNode("n_a", player=1, information_set=shared)
        n_b = DecisionNode("n_b", player=1, information_set=shared)

        r2_a_info = InformationSet("r2_a", player=1)
        r2_b_info = InformationSet("r2_b", player=1)
        deeper_a = DecisionNode("deeper_a", player=1, information_set=r2_a_info)
        deeper_b = DecisionNode("deeper_b", player=1, information_set=r2_b_info)

        deeper_a.add_child(TerminalNode("T1", (S(1), S(1))), "T1")
        deeper_a.add_child(TerminalNode("T2", (S(2), S(2))), "T2")
        deeper_b.add_child(TerminalNode("T4", (S(4), S(4))), "T4")
        deeper_b.add_child(TerminalNode("T5", (S(5), S(5))), "T5")

        n_a.add_child(deeper_a, "X")
        n_a.add_child(TerminalNode("T3", (S(3), S(3))), "Y")
        n_b.add_child(deeper_b, "X")
        n_b.add_child(TerminalNode("T6", (S(6), S(6))), "Y")

        root.add_child(n_a, "A")
        root.add_child(n_b, "B")

        p1_reduced = find_reduced_pure_strategies(root, player=1)

        assert len(p1_reduced) == 5
        assert Strategy({"shared": "Y"}) in p1_reduced
        assert Strategy({"shared": "X", "r2_a": "T1", "r2_b": "T4"}) in p1_reduced
        assert Strategy({"shared": "X", "r2_a": "T1", "r2_b": "T5"}) in p1_reduced
        assert Strategy({"shared": "X", "r2_a": "T2", "r2_b": "T4"}) in p1_reduced
        assert Strategy({"shared": "X", "r2_a": "T2", "r2_b": "T5"}) in p1_reduced


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

        # p0_full = find_full_pure_strategies(root, player=0)
        p0_reduced = [
            Strategy({"root": "L"}),
            Strategy({"n1": "a", "root": "R"}),
            Strategy({"n1": "b", "root": "R"}),
        ]

        reduced_array = create_payoff_array(root, [p0_reduced])

        assert reduced_array.shape == (3, 2)
