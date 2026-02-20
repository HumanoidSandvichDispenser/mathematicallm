"""
Tests for multiple equilibria support in backward induction.
"""

import pytest
import sympy as sp
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    TerminalNodeData,
    EquilibriumPath,
    ChanceNodeData,
)
import zermelo.services.subgame_perfect_equilibria as sge


def test_simple_two_player_game():
    """Test backward induction on a simple 2-player sequential game."""
    tree = GameTree(num_players=2)

    # Root: Player 0 decides
    tree.create_node(
        tag="P0 decision", identifier="root", data=DecisionNodeData(player=0)
    )

    # Player 0 chooses Left -> terminal with payoff (1, 0)
    tree.create_node(
        tag="Left outcome",
        identifier="left",
        parent="root",
        data=TerminalNodeData(payoffs=(1, 0)),
    )

    # Player 0 chooses Right -> Player 1 decides
    tree.create_node(
        tag="P1 decision",
        identifier="p1_node",
        parent="root",
        data=DecisionNodeData(player=1),
    )

    # Player 1 chooses Up -> (2, 2)
    tree.create_node(
        tag="Up outcome",
        identifier="up",
        parent="p1_node",
        data=TerminalNodeData(payoffs=(2, 2)),
    )

    # Player 1 chooses Down -> (0, 3)
    tree.create_node(
        tag="Down outcome",
        identifier="down",
        parent="p1_node",
        data=TerminalNodeData(payoffs=(0, 3)),
    )

    # Solve with backward induction
    result = sge.backward_induction(tree, mutate=True)

    # P1 will choose Down (payoff 3 > 2), so P0 gets 0
    # P0 compares Left (payoff 1) vs Right->Down (payoff 0)
    # P0 should choose Left, final payoff is (1, 0)
    assert result == [(sp.Integer(1), sp.Integer(0))]
    assert tree.get_node("root").data.bi_value == [(sp.Integer(1), sp.Integer(0))]


def test_chance_node():
    """Test backward induction with chance nodes."""
    tree = GameTree(num_players=2)

    # Root: chance node
    tree.create_node(tag="Nature", identifier="root", data=ChanceNodeData())

    # 50% chance: payoff (10, 0)
    tree.create_node(
        tag="Terminal 1",
        identifier="t1",
        parent="root",
        data=TerminalNodeData(payoffs=(10, 0), probability=sp.Rational(1, 2)),
    )

    # 50% chance: payoff (0, 10)
    tree.create_node(
        tag="Terminal 2",
        identifier="t2",
        parent="root",
        data=TerminalNodeData(payoffs=(0, 10), probability=sp.Rational(1, 2)),
    )

    # Solve
    result = sge.backward_induction(tree, mutate=True)

    # Expected value: 0.5 * (10, 0) + 0.5 * (0, 10) = (5, 5)
    assert result == [(sp.Integer(5), sp.Integer(5))]


def test_equilibria_with_sequential_tie():
    """Test game where tie is only in a subgame."""
    tree = GameTree(num_players=2)

    # Root: P0 decides
    tree.create_node("root", "root", data=DecisionNodeData(player=0))

    # P0 strictly prefers left
    tree.create_node("left", "left", parent="root", data=DecisionNodeData(player=1))
    tree.create_node(
        "right", "right", parent="root", data=TerminalNodeData(payoffs=(1, 0))
    )

    # Under left, P1 has a tie
    tree.create_node("L1", "L1", parent="left", data=TerminalNodeData(payoffs=(3, 2)))
    tree.create_node("L2", "L2", parent="left", data=TerminalNodeData(payoffs=(3, 2)))

    # Solve
    result = sge.backward_induction(tree, mutate=True)
    assert result == [(3, 2), (3, 2)]

    # Get equilibria
    equilibria = sge.get_all_equilibria(tree)

    assert len(equilibria) == 2, "Should have 2 equilibria (tie only under left)"

    actions_set = {frozenset(eq.actions.items()) for eq in equilibria}
    expected = {
        frozenset([("root", "left"), ("left", "L1")]),
        frozenset([("root", "left"), ("left", "L2")]),
    }
    assert actions_set == expected


def test_no_equilibria_without_bi():
    """Test that get_all_equilibria raises error if BI not run."""
    tree = GameTree(num_players=2)
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "left", "left", parent="root", data=TerminalNodeData(payoffs=(1, 0))
    )

    with pytest.raises(ValueError, match="backward_induction has not been run"):
        sge.get_all_equilibria(tree)


def test_symbolic_tie():
    """Test that symbolic expressions with equal values are detected as ties."""
    import sympy as sp

    tree = GameTree(num_players=1)

    x = sp.Symbol("x")
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    tree.create_node("A", "A", parent="root", data=TerminalNodeData(payoffs=(x,)))
    tree.create_node(
        "B", "B", parent="root", data=TerminalNodeData(payoffs=(x,))
    )  # Same value

    result = sge.backward_induction(tree, mutate=True)
    assert result == [(x,), (x,)]

    equilibria = sge.get_all_equilibria(tree)
    assert len(equilibria) == 2, "Symbolic tie should create 2 equilibria"


def test_tie_in_order():
    """Test that if two actions yield the same payoff for the current player but different payoffs for others, both are still considered equilibria."""
    tree1 = GameTree(num_players=2)
    tree1.create_node("root", "root", data=DecisionNodeData(player=0))
    tree1.create_node("A", "A", parent="root", data=DecisionNodeData(player=1))
    tree1.create_node("B", "B", parent="root", data=TerminalNodeData(payoffs=(4, 0)))
    tree1.create_node("C", "C", parent="A", data=TerminalNodeData(payoffs=(5, 0)))
    tree1.create_node("D", "D", parent="A", data=TerminalNodeData(payoffs=(3, 0)))

    tree2 = GameTree(num_players=2)
    tree2.create_node("root", "root", data=DecisionNodeData(player=0))
    tree2.create_node("A", "A", parent="root", data=DecisionNodeData(player=1))
    tree2.create_node("B", "B", parent="root", data=TerminalNodeData(payoffs=(4, 0)))
    tree2.create_node("D", "D", parent="A", data=TerminalNodeData(payoffs=(3, 0)))
    tree2.create_node("C", "C", parent="A", data=TerminalNodeData(payoffs=(5, 0)))

    sge.backward_induction(tree1, mutate=True)
    sge.backward_induction(tree2, mutate=True)

    print(tree1.show())
    print(tree2.show())

    print("SPNEs for tree1:")
    for spne in sge.get_all_spne(tree1):
        print(spne)

    print("SPNEs for tree2:")
    for spne in sge.get_all_spne(tree2):
        print(spne)


def test_tie_is_maintained():
    """Test that when a subgame has multiple equilibria, supergames will consider each of them when comparing equilibria"""

    tree = GameTree(num_players=2)

    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    tree.create_node("A", "A", parent="root", data=DecisionNodeData(player=1))
    tree.create_node("B", "B", parent="root", data=TerminalNodeData(payoffs=(4, 0)))
    tree.create_node("C", "C", parent="A", data=TerminalNodeData(payoffs=(3, 0)))
    tree.create_node("D", "D", parent="A", data=TerminalNodeData(payoffs=(5, 0)))
    # tree.create_node("root", "root", data=DecisionNodeData(player=0))
    # tree.create_node("A", "A", parent="root", data=DecisionNodeData(player=1))
    # tree.create_node("B", "B", parent="root", data=TerminalNodeData(payoffs=(4, 0)))
    # tree.create_node("C", "C", parent="A", data=DecisionNodeData(player=0))
    # tree.create_node("D", "D", parent="A", data=TerminalNodeData(payoffs=(5, 0)))
    # tree.create_node("E", "E", parent="C", data=TerminalNodeData(payoffs=(3, 0)))
    # tree.create_node("F", "F", parent="C", data=TerminalNodeData(payoffs=(5, 0)))

    sge.backward_induction(tree, mutate=True)

    equilibria = sge.get_all_equilibria(tree)

    # print(tree.show())

    # print("Equilibria found:")
    # for eq in equilibria:
    #    print(f"Actions: {eq.actions}, Payoffs: {eq.payoffs}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
