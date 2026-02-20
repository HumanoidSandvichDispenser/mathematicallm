"""
Tests for strategic form conversion with chance nodes.
"""

import pytest
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
)
from zermelo.services.strategic_form import (
    execute_strategy_profile,
    extensive_to_strategic,
)
from zermelo.services.strategy import find_full_pure_strategies
from zermelo.extensive.strategy import Strategy


def test_execute_with_chance_node_weighted_average():
    """
    Test that execute_strategy_profile correctly computes expected payoff
    at chance nodes using probability-weighted average.
    """
    tree = GameTree(num_players=2)

    tree.create_node("Chance", "chance", data=ChanceNodeData())
    tree.create_node(
        "Heads", "heads", parent="chance", data=TerminalNodeData(payoffs=(1, 0))
    )
    tree.create_node(
        "Tails", "tails", parent="chance", data=TerminalNodeData(payoffs=(0, 1))
    )

    # Set probabilities: 50% each (need to set via children)
    head_node = tree.get_node("heads")
    tail_node = tree.get_node("tails")
    head_node.data.probability = sp.Rational(1, 2)
    tail_node.data.probability = sp.Rational(1, 2)

    payoff = execute_strategy_profile(tree, {})

    # Expected: 0.5 * (1, 0) + 0.5 * (0, 1) = (0.5, 0.5)
    assert payoff == (sp.Rational(1, 2), sp.Rational(1, 2))


def test_execute_chance_with_decision_after():
    """
    Test chance node followed by a decision node.
    """
    import sympy as sp

    tree = GameTree(num_players=2)

    tree.create_node("Chance", "chance", data=ChanceNodeData())
    tree.create_node(
        "Heads -> P0", "heads", parent="chance", data=DecisionNodeData(player=0)
    )
    tree.create_node(
        "Tails -> P0", "tails", parent="chance", data=DecisionNodeData(player=0)
    )

    tree.create_node(
        "Heads Left", "h_l", parent="heads", data=TerminalNodeData(payoffs=(3, 1))
    )
    tree.create_node(
        "Heads Right", "h_r", parent="heads", data=TerminalNodeData(payoffs=(1, 2))
    )
    tree.create_node(
        "Tails Left", "t_l", parent="tails", data=TerminalNodeData(payoffs=(2, 0))
    )
    tree.create_node(
        "Tails Right", "t_r", parent="tails", data=TerminalNodeData(payoffs=(0, 3))
    )

    head_node = tree.get_node("heads")
    tail_node = tree.get_node("tails")
    head_node.data.probability = sp.Rational(1, 2)
    tail_node.data.probability = sp.Rational(1, 2)

    # P0 chooses Left at both chance outcomes
    profile = {
        0: Strategy({"heads": "h_l", "tails": "t_l"}),
    }

    payoff = execute_strategy_profile(tree, profile)

    # Expected: 0.5 * (3,1) + 0.5 * (2,0) = (2.5, 0.5)
    assert payoff == (sp.Rational(5, 2), sp.Rational(1, 2))


def test_extensive_to_strategic_with_chance():
    """
    Test strategic form conversion with chance nodes.
    """
    import sympy as sp

    tree = GameTree(num_players=2)

    tree.create_node("Chance", "chance", data=ChanceNodeData())
    tree.create_node(
        "Left", "left", parent="chance", data=TerminalNodeData(payoffs=(3, 0))
    )
    tree.create_node(
        "Right", "right", parent="chance", data=TerminalNodeData(payoffs=(0, 3))
    )

    head_node = tree.get_node("left")
    tail_node = tree.get_node("right")
    head_node.data.probability = sp.Rational(1, 2)
    tail_node.data.probability = sp.Rational(1, 2)

    strategies, payoffs = extensive_to_strategic(tree)

    # Both players have 1 strategy (no decisions)
    assert len(strategies[0]) == 1
    assert len(strategies[1]) == 1

    # Payoff should be expected value: 0.5*(3,0) + 0.5*(0,3) = (1.5, 1.5)
    assert payoffs[0, 0, 0] == sp.Rational(3, 2)
    assert payoffs[0, 0, 1] == sp.Rational(3, 2)


def test_chance_default_probability_one():
    """
    Test that missing probability defaults to 1.
    """
    tree = GameTree(num_players=2)

    tree.create_node("Chance", "chance", data=ChanceNodeData())
    tree.create_node(
        "Only child", "only", parent="chance", data=TerminalNodeData(payoffs=(5, 3))
    )
    # No probability set - should default to 1

    payoff = execute_strategy_profile(tree, {})

    assert payoff == (5, 3)


def test_nested_chance_nodes():
    """
    Test nested chance nodes compute expected value correctly.
    """
    import sympy as sp

    tree = GameTree(num_players=2)

    tree.create_node("Chance 1", "c1", data=ChanceNodeData())
    tree.create_node("c1_a", "c1_a", parent="c1", data=ChanceNodeData())
    tree.create_node("c1_b", "c1_b", parent="c1", data=ChanceNodeData())

    tree.create_node("a1", "a1", parent="c1_a", data=TerminalNodeData(payoffs=(10, 0)))
    tree.create_node("a2", "a2", parent="c1_a", data=TerminalNodeData(payoffs=(0, 10)))

    tree.create_node("b1", "b1", parent="c1_b", data=TerminalNodeData(payoffs=(5, 5)))
    # b2 doesn't exist, but we set probability for it as 0

    c1_a = tree.get_node("c1_a")
    c1_b = tree.get_node("c1_b")
    c1_a.data.probability = sp.Rational(1, 2)
    c1_b.data.probability = sp.Rational(1, 2)

    a1 = tree.get_node("a1")
    a2 = tree.get_node("a2")
    b1 = tree.get_node("b1")
    a1.data.probability = sp.Rational(1, 2)
    a2.data.probability = sp.Rational(1, 2)
    b1.data.probability = sp.Integer(1)  # only child, prob 1

    payoff = execute_strategy_profile(tree, {})

    # c1_a: 0.5 * (10,0) + 0.5 * (0,10) = (5,5)
    # c1_b: 1 * (5,5) = (5,5)
    # c1: 0.5 * (5,5) + 0.5 * (5,5) = (5,5)
    assert payoff == (5, 5)


import sympy as sp
