"""
Pytest fixtures for strategy service tests.
"""

import pytest
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    TerminalNodeData,
)


@pytest.fixture
def simple_perfect_info_tree():
    """Simple perfect information game: P0 decides at root."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "Left", "left", parent="root", data=TerminalNodeData(payoffs=(1, 0))
    )
    tree.create_node(
        "Right", "right", parent="root", data=TerminalNodeData(payoffs=(0, 1))
    )
    return tree


@pytest.fixture
def two_decisions_per_player_tree():
    """Perfect information game with 2 decisions per player."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node("P1 left", "p1_l", parent="root", data=DecisionNodeData(player=1))
    tree.create_node("P1 right", "p1_r", parent="root", data=DecisionNodeData(player=1))
    tree.create_node("T1", "t1", parent="p1_l", data=TerminalNodeData(payoffs=(1, 1)))
    tree.create_node("T2", "t2", parent="p1_l", data=TerminalNodeData(payoffs=(2, 0)))
    tree.create_node("T3", "t3", parent="p1_r", data=TerminalNodeData(payoffs=(0, 2)))
    tree.create_node("T4", "t4", parent="p1_r", data=TerminalNodeData(payoffs=(3, 3)))
    return tree


@pytest.fixture
def imperfect_info_same_actions_tree():
    """Imperfect information game where P1 has same actions at both info set nodes."""
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "P1 at left",
        "p1_l",
        parent="root",
        data=DecisionNodeData(player=1, information_set="I1"),
    )
    tree.create_node(
        "P1 at right",
        "p1_r",
        parent="root",
        data=DecisionNodeData(player=1, information_set="I1"),
    )
    tree.create_node(
        "Up L", "up_l", parent="p1_l", data=TerminalNodeData(payoffs=(1, 1))
    )
    tree.create_node(
        "Down L", "down_l", parent="p1_l", data=TerminalNodeData(payoffs=(2, 0))
    )
    tree.create_node(
        "Up R", "up_r", parent="p1_r", data=TerminalNodeData(payoffs=(0, 2))
    )
    tree.create_node(
        "Down R", "down_r", parent="p1_r", data=TerminalNodeData(payoffs=(3, 3))
    )
    return tree


@pytest.fixture
def escalation_game():
    """
    The Escalation Game (perfect information, 2 players).

    Player 1 moves first:
      - Accept -> (0, 0)
      - Threaten -> Player 2 moves:
          - Concede -> (1, -2)
          - Escalate -> Player 1 moves:
              - Give Up -> (-2, 1)
              - War     -> (-1, -1)
    """
    tree = GameTree(num_players=2)
    tree.create_node("Player 1", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "Accept", "accept", parent="root", data=TerminalNodeData(payoffs=(0, 0))
    )
    tree.create_node(
        "Player 2", "threaten", parent="root", data=DecisionNodeData(player=1)
    )
    tree.create_node(
        "Concede", "concede", parent="threaten", data=TerminalNodeData(payoffs=(1, -2))
    )
    tree.create_node(
        "Player 1 again", "escalate", parent="threaten", data=DecisionNodeData(player=0)
    )
    tree.create_node(
        "Give Up", "give_up", parent="escalate", data=TerminalNodeData(payoffs=(-2, 1))
    )
    tree.create_node(
        "War", "war", parent="escalate", data=TerminalNodeData(payoffs=(-1, -1))
    )
    return tree
