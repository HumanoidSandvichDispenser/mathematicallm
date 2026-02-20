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
    assert result == (sp.Integer(1), sp.Integer(0))
    assert tree.get_node("root").data.bi_value == (sp.Integer(1), sp.Integer(0))


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
    assert result == (sp.Integer(5), sp.Integer(5))


def test_backward_induction_imperfect_information():
    """Test that backward induction raises NotImplementedError for imperfect information."""
    tree = GameTree(num_players=2)
    tree.create_node(
        "P0 A", "n1", data=DecisionNodeData(player=0, information_set="I0")
    )
    tree.create_node(
        "P0 B", "n2", parent="n1", data=DecisionNodeData(player=0, information_set="I0")
    )
    tree.create_node(
        "P0 C", "n3", parent="n1", data=DecisionNodeData(player=0, information_set="I0")
    )
    tree.create_node(
        "Terminal", "t1", parent="n2", data=TerminalNodeData(payoffs=(1, 0))
    )

    with pytest.raises(NotImplementedError) as exc_info:
        sge.backward_induction(tree, mutate=True)
    assert "imperfect information" in str(exc_info.value).lower()


def test_symbolic_payoffs():
    """Test backward induction with symbolic payoffs."""
    x = sp.Symbol("x")

    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))

    tree.create_node(
        "Left", "left", parent="root", data=TerminalNodeData(payoffs=(x, 0))
    )

    tree.create_node(
        "Right", "right", parent="root", data=TerminalNodeData(payoffs=(x + 1, 1))
    )

    result = sge.backward_induction(tree, mutate=True)

    # P0 chooses max(x, x+1) which is x+1
    assert result[0] == x + 1
    assert result[1] == sp.Integer(1)


def test_unique_equilibrium():
    """Test game with a unique equilibrium (no ties)."""
    tree = GameTree(num_players=2)
    
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    tree.create_node("left", "left", parent="root", data=TerminalNodeData(payoffs=(3, 1)))
    tree.create_node("right", "right", parent="root", data=TerminalNodeData(payoffs=(1, 2)))
    
    # Solve
    result = sge.backward_induction(tree, mutate=True)
    assert result == (3, 1), "P0 should choose left"
    
    # Get equilibria
    equilibria = sge.get_all_equilibria(tree)
    
    assert len(equilibria) == 1, "Should have exactly one equilibrium"
    eq = equilibria[0]
    assert eq.payoffs == (3, 1)
    assert eq.actions == {"root": "left"}


def test_two_equilibria_one_decision_node():
    """Test game with two equilibria due to tie at root."""
    tree = GameTree(num_players=2)
    
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    tree.create_node("left", "left", parent="root", data=TerminalNodeData(payoffs=(2, 1)))
    tree.create_node("right", "right", parent="root", data=TerminalNodeData(payoffs=(2, 3)))
    
    # Solve
    result = sge.backward_induction(tree, mutate=True)
    assert result == (2, 1) or result == (2, 3), "Both give P0 payoff of 2"
    
    # Check optimal children stored
    root = tree.get_node("root")
    assert set(root.data.optimal_children) == {"left", "right"}, "Both children should be optimal"
    
    # Get equilibria
    equilibria = sge.get_all_equilibria(tree)
    
    assert len(equilibria) == 2, "Should have exactly two equilibria"
    
    # Extract actions from equilibria
    actions_set = {frozenset(eq.actions.items()) for eq in equilibria}
    expected = {
        frozenset([("root", "left")]),
        frozenset([("root", "right")])
    }
    assert actions_set == expected
    
    # All should have same payoff
    for eq in equilibria:
        assert eq.payoffs == (2, 1) or eq.payoffs == (2, 3)


def test_four_equilibria_two_decision_nodes():
    """Test game with 4 equilibria (2x2 from independent ties)."""
    tree = GameTree(num_players=2)
    
    # Root: P0 decides
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    
    # P0 has two equally good choices
    tree.create_node("A", "A", parent="root", data=DecisionNodeData(player=1))
    tree.create_node("B", "B", parent="root", data=DecisionNodeData(player=1))
    
    # Under A, P1 has two equally good choices
    tree.create_node("A1", "A1", parent="A", data=TerminalNodeData(payoffs=(3, 2)))
    tree.create_node("A2", "A2", parent="A", data=TerminalNodeData(payoffs=(3, 2)))
    
    # Under B, P1 has two equally good choices
    tree.create_node("B1", "B1", parent="B", data=TerminalNodeData(payoffs=(3, 2)))
    tree.create_node("B2", "B2", parent="B", data=TerminalNodeData(payoffs=(3, 2)))
    
    # Solve
    result = sge.backward_induction(tree, mutate=True)
    assert result == (3, 2)
    
    # Get equilibria
    equilibria = sge.get_all_equilibria(tree)
    
    assert len(equilibria) == 4, "Should have 2x2 = 4 equilibria"
    
    # Check that all combinations are present
    actions_set = {frozenset(eq.actions.items()) for eq in equilibria}
    expected = {
        frozenset([("root", "A"), ("A", "A1")]),
        frozenset([("root", "A"), ("A", "A2")]),
        frozenset([("root", "B"), ("B", "B1")]),
        frozenset([("root", "B"), ("B", "B2")]),
    }
    assert actions_set == expected
    
    # All should have same payoff
    for eq in equilibria:
        assert eq.payoffs == (3, 2)


def test_equilibria_with_sequential_tie():
    """Test game where tie is only in a subgame."""
    tree = GameTree(num_players=2)
    
    # Root: P0 decides
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    
    # P0 strictly prefers left
    tree.create_node("left", "left", parent="root", data=DecisionNodeData(player=1))
    tree.create_node("right", "right", parent="root", data=TerminalNodeData(payoffs=(1, 0)))
    
    # Under left, P1 has a tie
    tree.create_node("L1", "L1", parent="left", data=TerminalNodeData(payoffs=(3, 2)))
    tree.create_node("L2", "L2", parent="left", data=TerminalNodeData(payoffs=(3, 2)))
    
    # Solve
    result = sge.backward_induction(tree, mutate=True)
    assert result == (3, 2)
    
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
    tree.create_node("left", "left", parent="root", data=TerminalNodeData(payoffs=(1, 0)))
    
    with pytest.raises(ValueError, match="backward_induction has not been run"):
        sge.get_all_equilibria(tree)


def test_symbolic_tie():
    """Test that symbolic expressions with equal values are detected as ties."""
    import sympy as sp
    
    tree = GameTree(num_players=1)
    
    x = sp.Symbol('x')
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    tree.create_node("A", "A", parent="root", data=TerminalNodeData(payoffs=(x,)))
    tree.create_node("B", "B", parent="root", data=TerminalNodeData(payoffs=(x,)))  # Same value
    
    result = sge.backward_induction(tree, mutate=True)
    assert result == (x,)
    
    equilibria = sge.get_all_equilibria(tree)
    assert len(equilibria) == 2, "Symbolic tie should create 2 equilibria"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
