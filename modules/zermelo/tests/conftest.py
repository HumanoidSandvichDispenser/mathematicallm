"""
Pytest fixtures for strategy service tests.
"""

#import pytest
#from zermelo.extensive import (
#    GameTree,
#    DecisionNodeData,
#    TerminalNodeData,
#)
#
#
#@pytest.fixture
#def simple_perfect_info_tree():
#    """Simple perfect information game: P0 decides at root."""
#    tree = GameTree(num_players=2)
#    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
#    tree.create_node(
#        "Left", "left", parent="root", data=TerminalNodeData(payoffs=(1, 0))
#    )
#    tree.create_node(
#        "Right", "right", parent="root", data=TerminalNodeData(payoffs=(0, 1))
#    )
#    return tree
#
#
#@pytest.fixture
#def two_decisions_per_player_tree():
#    """Perfect information game with 2 decisions per player."""
#    tree = GameTree(num_players=2)
#    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
#    tree.create_node("P1 left", "p1_l", parent="root", data=DecisionNodeData(player=1))
#    tree.create_node("P1 right", "p1_r", parent="root", data=DecisionNodeData(player=1))
#    tree.create_node("T1", "t1", parent="p1_l", data=TerminalNodeData(payoffs=(1, 1)))
#    tree.create_node("T2", "t2", parent="p1_l", data=TerminalNodeData(payoffs=(2, 0)))
#    tree.create_node("T3", "t3", parent="p1_r", data=TerminalNodeData(payoffs=(0, 2)))
#    tree.create_node("T4", "t4", parent="p1_r", data=TerminalNodeData(payoffs=(3, 3)))
#    return tree
#
#
#@pytest.fixture
#def imperfect_info_same_actions_tree():
#    """Imperfect information game where P1 has same actions at both info set nodes."""
#    tree = GameTree(num_players=2)
#    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
#    tree.create_node(
#        "P1 at left",
#        "p1_l",
#        parent="root",
#        data=DecisionNodeData(player=1, information_set="I1", actions=["up", "down"]),
#    )
#    tree.create_node(
#        "P1 at right",
#        "p1_r",
#        parent="root",
#        data=DecisionNodeData(player=1, information_set="I1", actions=["up", "down"]),
#    )
#    tree.create_node(
#        "Up L", "up_l", parent="p1_l", data=TerminalNodeData(payoffs=(1, 1))
#    )
#    tree.create_node(
#        "Down L", "down_l", parent="p1_l", data=TerminalNodeData(payoffs=(2, 0))
#    )
#    tree.create_node(
#        "Up R", "up_r", parent="p1_r", data=TerminalNodeData(payoffs=(0, 2))
#    )
#    tree.create_node(
#        "Down R", "down_r", parent="p1_r", data=TerminalNodeData(payoffs=(3, 3))
#    )
#    return tree
#
#
#@pytest.fixture
#def imperfect_info_with_extra_decision_tree():
#    """
#    Imperfect information game where P1 has an info set I1, but one branch
#    leads to another P0 decision which then gives P1 another info set I2.
#    
#    Tree structure:
#        Root (P0)
#         /     \\
#      p1_l     p1_r       <- P1 info set I1, actions: ["up", "down"]
#      /   \\      /  \\
#   up_l down_l up_r down_r
#    |      |      |     |
#   term  p0_c   term  term   <- p0_cont continues game when P1 picks "down"
#           |
#         P1 (I2)              <- P1 info set I2 (only reachable via "down" at p1_l)
#         /   \\
#        x     y
#        |     |
#       t_x   t_y
#    """
#    tree = GameTree(num_players=2)
#    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
#
#    # P1 info set I1 with two nodes
#    tree.create_node(
#        "P1 at left",
#        "p1_l",
#        parent="root",
#        data=DecisionNodeData(player=1, information_set="I1", actions=["up", "down"]),
#    )
#    tree.create_node(
#        "P1 at right",
#        "p1_r",
#        parent="root",
#        data=DecisionNodeData(player=1, information_set="I1", actions=["up", "down"]),
#    )
#
#    # P1 at left: up leads to terminal, down leads to P0 continuation
#    tree.create_node(
#        "Up L", "up_l", parent="p1_l", data=TerminalNodeData(payoffs=(1, 1))
#    )
#    tree.create_node(
#        "P0 cont", "p0_cont", parent="p1_l", data=DecisionNodeData(player=0)
#    )
#
#    # P1 at right: both lead to terminals
#    tree.create_node(
#        "Up R", "up_r", parent="p1_r", data=TerminalNodeData(payoffs=(0, 2))
#    )
#    tree.create_node(
#        "Down R", "down_r", parent="p1_r", data=TerminalNodeData(payoffs=(3, 3))
#    )
#
#    # P0 continuation node leads to P1's second info set I2
#    tree.create_node(
#        "P1 at I2",
#        "p1_i2",
#        parent="p0_cont",
#        data=DecisionNodeData(player=1, information_set="I2", actions=["x", "y"]),
#    )
#
#    # I2 terminals
#    tree.create_node("X", "x", parent="p1_i2", data=TerminalNodeData(payoffs=(5, 5)))
#    tree.create_node("Y", "y", parent="p1_i2", data=TerminalNodeData(payoffs=(6, 4)))
#
#    return tree
#
#
#@pytest.fixture
#def prisoners_dilemma_tree():
#    """
#    Prisoner's Dilemma as an extensive-form game.
#
#    P0 moves first (simultaneous: P1 doesn't see P0's choice).
#    P1 has an info set I1 with two nodes representing the same decision.
#
#    Payoff matrix (P0, P1):
#                  P1 Cooperate P1 Defect
#    P0 Cooperate  (3, 3)       (0, 5)
#    P0 Defect     (5, 0)       (1, 1)
#    """
#    tree = GameTree(num_players=2)
#
#    # P0 moves first (doesn't matter what P0 does, P1 doesn't see it)
#    tree.create_node(
#        "P0", "root", data=DecisionNodeData(player=0, actions=["cooperate", "defect"])
#    )
#
#    # P0 chooses cooperate
#    tree.create_node(
#        "P0 Cooperate",
#        "p0_c",
#        parent="root",
#        data=DecisionNodeData(
#            player=1, information_set="I1", actions=["cooperate", "defect"]
#        ),
#    )
#
#    # P0 chooses defect
#    tree.create_node(
#        "P0 Defect",
#        "p0_d",
#        parent="root",
#        data=DecisionNodeData(
#            player=1, information_set="I1", actions=["cooperate", "defect"]
#        ),
#    )
#
#    # From P0 Cooperate: P1 Cooperate -> (3, 3)
#    tree.create_node(
#        "P1 Cooperate",
#        "p1_c_from_c",
#        parent="p0_c",
#        data=TerminalNodeData(payoffs=(3, 3)),
#    )
#    # From P0 Cooperate: P1 Defect -> (0, 5)
#    tree.create_node(
#        "P1 Defect", "p1_d_from_c", parent="p0_c", data=TerminalNodeData(payoffs=(0, 5))
#    )
#
#    # From P0 Defect: P1 Cooperate -> (5, 0)
#    tree.create_node(
#        "P1 Cooperate",
#        "p1_c_from_d",
#        parent="p0_d",
#        data=TerminalNodeData(payoffs=(5, 0)),
#    )
#    # From P0 Defect: P1 Defect -> (1, 1)
#    tree.create_node(
#        "P1 Defect", "p1_d_from_d", parent="p0_d", data=TerminalNodeData(payoffs=(1, 1))
#    )
#
#    return tree
#
#
#@pytest.fixture
#def escalation_game():
#    """
#    The Escalation Game (perfect information, 2 players).
#
#    Player 1 moves first:
#      - Accept -> (0, 0)
#      - Threaten -> Player 2 moves:
#          - Concede -> (1, -2)
#          - Escalate -> Player 1 moves:
#              - Give Up -> (-2, 1)
#              - War     -> (-1, -1)
#    """
#    tree = GameTree(num_players=2)
#    tree.create_node("Player 1", "root", data=DecisionNodeData(player=0))
#    tree.create_node(
#        "Accept", "accept", parent="root", data=TerminalNodeData(payoffs=(0, 0))
#    )
#    tree.create_node(
#        "Player 2", "threaten", parent="root", data=DecisionNodeData(player=1)
#    )
#    tree.create_node(
#        "Concede", "concede", parent="threaten", data=TerminalNodeData(payoffs=(1, -2))
#    )
#    tree.create_node(
#        "Player 1 again", "escalate", parent="threaten", data=DecisionNodeData(player=0)
#    )
#    tree.create_node(
#        "Give Up", "give_up", parent="escalate", data=TerminalNodeData(payoffs=(-2, 1))
#    )
#    tree.create_node(
#        "War", "war", parent="escalate", data=TerminalNodeData(payoffs=(-1, -1))
#    )
#    return tree
