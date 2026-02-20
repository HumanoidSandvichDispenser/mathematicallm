"""
Tests for strategy service - finding full pure strategies.
"""

import pytest
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    TerminalNodeData,
)
from zermelo.services.strategy_service import find_full_pure_strategies


def test_perfect_information_single_decision(simple_perfect_info_tree):
    """Test perfect information game where player has single decision node."""
    strategies = find_full_pure_strategies(simple_perfect_info_tree, player=0)

    assert len(strategies) == 2
    decision_sets = [s.decisions for s in strategies]
    assert {"root": "left"} in decision_sets
    assert {"root": "right"} in decision_sets


def test_perfect_information_two_players_two_decisions_each(
    two_decisions_per_player_tree,
):
    """Test perfect information game with 2 decisions per player."""
    p0_strategies = find_full_pure_strategies(two_decisions_per_player_tree, player=0)
    p1_strategies = find_full_pure_strategies(two_decisions_per_player_tree, player=1)

    assert len(p0_strategies) == 2
    assert len(p1_strategies) == 4


def test_perfect_information_no_decisions():
    """Test player with no decision nodes returns single empty strategy."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "Terminal", "t", parent="root", data=TerminalNodeData(payoffs=(1, 1))
    )

    strategies = find_full_pure_strategies(tree, player=1)

    assert len(strategies) == 1
    assert strategies.pop().decisions == {}


def test_imperfect_information_same_actions(imperfect_info_same_actions_tree):
    """Test imperfect information where player has same actions at both info set nodes."""
    strategies = find_full_pure_strategies(imperfect_info_same_actions_tree, player=1)

    assert len(strategies) == 2
    decision_sets = [s.decisions for s in strategies]
    assert {"I1": "up_l"} in decision_sets
    assert {"I1": "down_l"} in decision_sets
    assert {"I1": "up_r"} not in decision_sets
    assert {"I1": "down_r"} not in decision_sets


def test_imperfect_information_multiple_info_sets():
    """Test imperfect information with multiple info sets for the same player."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node("P1 node", "p1", parent="root", data=DecisionNodeData(player=1))
    tree.create_node(
        "P0_2a",
        "p0_2a",
        parent="p1",
        data=DecisionNodeData(player=0, information_set="I0"),
    )
    tree.create_node(
        "P0_2b",
        "p0_2b",
        parent="p1",
        data=DecisionNodeData(player=0, information_set="I0"),
    )
    tree.create_node("T1", "t1", parent="p0_2a", data=TerminalNodeData(payoffs=(1, 1)))
    tree.create_node("T2", "t2", parent="p0_2a", data=TerminalNodeData(payoffs=(2, 0)))
    tree.create_node("T3", "t3", parent="p0_2b", data=TerminalNodeData(payoffs=(0, 2)))
    tree.create_node("T4", "t4", parent="p0_2b", data=TerminalNodeData(payoffs=(3, 3)))

    p0_strategies = find_full_pure_strategies(tree, player=0)
    p1_strategies = find_full_pure_strategies(tree, player=1)

    assert len(p0_strategies) == 2
    assert len(p1_strategies) == 2


def test_three_player_game():
    """Test with three players."""
    tree = GameTree(num_players=3)
    tree.create_node("P0", "root", data=DecisionNodeData(player=0))
    tree.create_node("P1", "p1", parent="root", data=DecisionNodeData(player=1))
    tree.create_node("P2", "p2", parent="p1", data=DecisionNodeData(player=2))
    tree.create_node("T", "t", parent="p2", data=TerminalNodeData(payoffs=(1, 2, 3)))

    assert len(find_full_pure_strategies(tree, player=0)) == 1
    assert len(find_full_pure_strategies(tree, player=1)) == 1
    assert len(find_full_pure_strategies(tree, player=2)) == 1


def test_strategy_preserves_info_set_ids():
    """Test that returned strategies use correct info set identifiers."""
    tree = GameTree(num_players=2)
    tree.create_node(
        "Root", "root", data=DecisionNodeData(player=0, information_set="root_info")
    )
    tree.create_node(
        "P1a",
        "p1a",
        parent="root",
        data=DecisionNodeData(player=1, information_set="p1_info"),
    )
    tree.create_node(
        "P1b",
        "p1b",
        parent="root",
        data=DecisionNodeData(player=1, information_set="p1_info"),
    )
    tree.create_node("T", "t", parent="p1a", data=TerminalNodeData(payoffs=(1, 1)))

    strategies = find_full_pure_strategies(tree, player=1)

    assert len(strategies) == 1
    strategy = list(strategies)[0]
    assert "p1_info" in strategy.decisions
    assert strategy.decisions["p1_info"] == "t"


def test_default_info_set_uses_node_id():
    """Test that default info set (when not specified) uses node ID."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node("Child", "child", parent="root", data=DecisionNodeData(player=1))
    tree.create_node("T1", "t1", parent="child", data=TerminalNodeData(payoffs=(1, 1)))
    tree.create_node("T2", "t2", parent="child", data=TerminalNodeData(payoffs=(2, 0)))

    strategies = find_full_pure_strategies(tree, player=1)

    assert len(strategies) == 2
    decision_sets = [s.decisions for s in strategies]
    assert {"child": "t1"} in decision_sets
    assert {"child": "t2"} in decision_sets


def test_mixed_info_sets_and_no_info_sets():
    """Test game where some nodes have explicit info sets, some don't."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "P1a",
        "p1a",
        parent="root",
        data=DecisionNodeData(player=1, information_set="same"),
    )
    tree.create_node(
        "P1b",
        "p1b",
        parent="root",
        data=DecisionNodeData(player=1, information_set="same"),
    )
    tree.create_node("T1", "t1", parent="p1a", data=TerminalNodeData(payoffs=(1, 1)))
    tree.create_node("T2", "t2", parent="p1a", data=TerminalNodeData(payoffs=(2, 0)))
    tree.create_node("T3", "t3", parent="p1b", data=TerminalNodeData(payoffs=(3, 3)))
    tree.create_node("T4", "t4", parent="p1b", data=TerminalNodeData(payoffs=(4, 2)))

    strategies = find_full_pure_strategies(tree, player=1)

    assert len(strategies) == 2
    decision_sets = [s.decisions for s in strategies]
    assert {"same": "t1"} in decision_sets
    assert {"same": "t2"} in decision_sets


def test_p0_strategies_in_imperfect_game(imperfect_info_same_actions_tree):
    """Test player 0 strategies in imperfect information game."""
    strategies = find_full_pure_strategies(imperfect_info_same_actions_tree, player=0)

    assert len(strategies) == 2
    decision_sets = [s.decisions for s in strategies]
    assert {"root": "p1_l"} in decision_sets
    assert {"root": "p1_r"} in decision_sets
