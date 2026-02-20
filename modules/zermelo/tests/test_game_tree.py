"""
Tests for extensive-form game tree implementation.
"""

import pytest
import sympy as sp
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
)


def test_simple_two_player_game():
    """Test backward induction on a simple 2-player sequential game."""
    tree = GameTree(num_players=2)
    
    # Root: Player 0 decides
    tree.create_node(
        tag="P0 decision",
        identifier="root",
        data=DecisionNodeData(player=0)
    )
    
    # Player 0 chooses Left -> terminal with payoff (1, 0)
    tree.create_node(
        tag="Left outcome",
        identifier="left",
        parent="root",
        data=TerminalNodeData(payoffs=(1, 0))
    )
    
    # Player 0 chooses Right -> Player 1 decides
    tree.create_node(
        tag="P1 decision",
        identifier="p1_node",
        parent="root",
        data=DecisionNodeData(player=1)
    )
    
    # Player 1 chooses Up -> (2, 2)
    tree.create_node(
        tag="Up outcome",
        identifier="up",
        parent="p1_node",
        data=TerminalNodeData(payoffs=(2, 2))
    )
    
    # Player 1 chooses Down -> (0, 3)
    tree.create_node(
        tag="Down outcome",
        identifier="down",
        parent="p1_node",
        data=TerminalNodeData(payoffs=(0, 3))
    )
    
    # Solve with backward induction
    result = tree.backward_induction(mutate=True)
    
    # P1 will choose Down (payoff 3 > 2), so P0 gets 0
    # P0 compares Left (payoff 1) vs Right->Down (payoff 0)
    # P0 should choose Left, final payoff is (1, 0)
    assert result == (sp.Integer(1), sp.Integer(0))
    assert tree.get_node("root").data.bi_value == (sp.Integer(1), sp.Integer(0))


def test_chance_node():
    """Test backward induction with chance nodes."""
    tree = GameTree(num_players=2)
    
    # Root: chance node
    tree.create_node(
        tag="Nature",
        identifier="root",
        data=ChanceNodeData()
    )
    
    # 50% chance: payoff (10, 0)
    tree.create_node(
        tag="Terminal 1",
        identifier="t1",
        parent="root",
        data=TerminalNodeData(payoffs=(10, 0), probability=sp.Rational(1, 2))
    )
    
    # 50% chance: payoff (0, 10)
    tree.create_node(
        tag="Terminal 2",
        identifier="t2",
        parent="root",
        data=TerminalNodeData(payoffs=(0, 10), probability=sp.Rational(1, 2))
    )
    
    # Solve
    result = tree.backward_induction(mutate=True)
    
    # Expected value: 0.5 * (10, 0) + 0.5 * (0, 10) = (5, 5)
    assert result == (sp.Integer(5), sp.Integer(5))


def test_symbolic_payoffs():
    """Test backward induction with symbolic payoffs."""
    x = sp.Symbol('x')
    
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    
    tree.create_node(
        "Left",
        "left",
        parent="root",
        data=TerminalNodeData(payoffs=(x, 0))
    )
    
    tree.create_node(
        "Right",
        "right",
        parent="root",
        data=TerminalNodeData(payoffs=(x + 1, 1))
    )
    
    result = tree.backward_induction(mutate=True)
    
    # P0 chooses max(x, x+1) which is x+1
    assert result[0] == x + 1
    assert result[1] == sp.Integer(1)


def test_serialization_round_trip():
    """Test that serialization and deserialization preserves the tree."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "Left",
        "left",
        parent="root",
        data=TerminalNodeData(payoffs=(1, 0))
    )
    
    # Serialize
    data = tree.to_dict()
    
    # Deserialize
    tree2 = GameTree.from_dict(data)
    
    # Verify structure
    assert tree2.root == "root"
    assert len(tree2.all_nodes()) == 2
    
    # Verify node data
    root_node = tree2.get_node("root")
    assert isinstance(root_node.data, DecisionNodeData)
    assert root_node.data.player == 0
    
    left_node = tree2.get_node("left")
    assert isinstance(left_node.data, TerminalNodeData)
    assert left_node.data.payoffs == (sp.Integer(1), sp.Integer(0))


def test_serialization_with_bi_values():
    """Test serialization preserves backward induction results."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node("Left", "left", parent="root", data=TerminalNodeData(payoffs=(1, 0)))
    tree.create_node("Right", "right", parent="root", data=TerminalNodeData(payoffs=(2, 1)))
    
    # Solve
    tree.backward_induction(mutate=True)
    
    # Serialize
    data = tree.to_dict()
    
    # Deserialize
    tree2 = GameTree.from_dict(data)
    
    # Verify BI values preserved
    root = tree2.get_node("root")
    assert root.data.bi_value == (sp.Integer(2), sp.Integer(1))


def test_node_type_checks():
    """Test GameNode type check properties."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node("Decision", "d1", parent="root", data=DecisionNodeData(player=1))
    tree.create_node("Chance", "c1", parent="root", data=ChanceNodeData(probability=sp.Rational(1, 2)))
    tree.create_node("Terminal", "t1", parent="root", data=TerminalNodeData(payoffs=(0,)))
    
    d_node = tree.get_node("d1")
    c_node = tree.get_node("c1")
    t_node = tree.get_node("t1")
    
    assert d_node.is_decision
    assert not d_node.is_chance
    assert not d_node.is_terminal
    
    assert c_node.is_chance
    assert not c_node.is_decision
    assert not c_node.is_terminal
    
    assert t_node.is_terminal
    assert not t_node.is_decision
    assert not t_node.is_chance


def test_probability_sympify():
    """Test that probabilities are automatically sympified."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=ChanceNodeData())
    
    # Pass probabilities as different types
    tree.create_node("A", "a", parent="root", data=TerminalNodeData(payoffs=(1,), probability=0.5))
    tree.create_node("B", "b", parent="root", data=TerminalNodeData(payoffs=(2,), probability="1/2"))
    tree.create_node("C", "c", parent="root", data=TerminalNodeData(payoffs=(3,), probability=sp.Rational(1, 2)))
    
    # All should be sympy expressions
    assert isinstance(tree.get_node("a").data.probability, sp.Expr)
    assert isinstance(tree.get_node("b").data.probability, sp.Expr)
    assert isinstance(tree.get_node("c").data.probability, sp.Expr)


def test_payoffs_sympify():
    """Test that payoffs are automatically sympified."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    
    # Pass payoffs as different types
    tree.create_node("A", "a", parent="root", data=TerminalNodeData(payoffs=(1, 2)))
    tree.create_node("B", "b", parent="root", data=TerminalNodeData(payoffs=(1.5, "3/2")))
    
    a_node = tree.get_node("a")
    assert all(isinstance(p, sp.Expr) for p in a_node.data.payoffs)
    assert a_node.data.payoffs == (sp.Integer(1), sp.Integer(2))
    
    b_node = tree.get_node("b")
    assert all(isinstance(p, sp.Expr) for p in b_node.data.payoffs)
