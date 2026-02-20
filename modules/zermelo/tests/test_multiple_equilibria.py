"""
Tests for multiple equilibria support in backward induction.
"""

import pytest
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    TerminalNodeData,
    EquilibriumPath,
)


def test_unique_equilibrium():
    """Test game with a unique equilibrium (no ties)."""
    tree = GameTree(num_players=2)
    
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    tree.create_node("left", "left", parent="root", data=TerminalNodeData(payoffs=(3, 1)))
    tree.create_node("right", "right", parent="root", data=TerminalNodeData(payoffs=(1, 2)))
    
    # Solve
    result = tree.backward_induction(mutate=True)
    assert result == (3, 1), "P0 should choose left"
    
    # Get equilibria
    equilibria = tree.get_all_equilibria()
    
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
    result = tree.backward_induction(mutate=True)
    assert result == (2, 1) or result == (2, 3), "Both give P0 payoff of 2"
    
    # Check optimal children stored
    root = tree.get_node("root")
    assert set(root.data.optimal_children) == {"left", "right"}, "Both children should be optimal"
    
    # Get equilibria
    equilibria = tree.get_all_equilibria()
    
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
    result = tree.backward_induction(mutate=True)
    assert result == (3, 2)
    
    # Get equilibria
    equilibria = tree.get_all_equilibria()
    
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
    result = tree.backward_induction(mutate=True)
    assert result == (3, 2)
    
    # Get equilibria
    equilibria = tree.get_all_equilibria()
    
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
        tree.get_all_equilibria()


def test_symbolic_tie():
    """Test that symbolic expressions with equal values are detected as ties."""
    import sympy as sp
    
    tree = GameTree(num_players=1)
    
    x = sp.Symbol('x')
    tree.create_node("root", "root", data=DecisionNodeData(player=0))
    tree.create_node("A", "A", parent="root", data=TerminalNodeData(payoffs=(x,)))
    tree.create_node("B", "B", parent="root", data=TerminalNodeData(payoffs=(x,)))  # Same value
    
    result = tree.backward_induction(mutate=True)
    assert result == (x,)
    
    equilibria = tree.get_all_equilibria()
    assert len(equilibria) == 2, "Symbolic tie should create 2 equilibria"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
