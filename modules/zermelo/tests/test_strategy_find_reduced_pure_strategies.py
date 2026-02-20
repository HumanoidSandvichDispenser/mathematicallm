"""
Tests for strategy_service.find_reduced_pure_strategies.

Reduced pure strategies only specify actions for information sets that are
actually reachable given the player's own earlier choices.  Info sets that
are cut off by the player's own decisions are omitted.
"""

import pytest
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    TerminalNodeData,
)
from zermelo.services.strategy import find_reduced_pure_strategies
from zermelo.extensive.strategy import Strategy


# ---------------------------------------------------------------------------
# Escalation game (perfect information, 2 players)
# ---------------------------------------------------------------------------


def test_escalation_game_p0_count(escalation_game):
    """Player 0 has exactly 3 reduced strategies in the escalation game."""
    strategies = find_reduced_pure_strategies(escalation_game, player=0)
    assert len(strategies) == 3


def test_escalation_game_p0_strategies(escalation_game):
    """Player 0's reduced strategies are correct."""
    strategies = find_reduced_pure_strategies(escalation_game, player=0)
    expected = {
        # Accept at root — escalate is unreachable, so not specified
        Strategy({"root": "accept"}),
        # Threaten, then Give Up
        Strategy({"root": "threaten", "escalate": "give_up"}),
        # Threaten, then War
        Strategy({"root": "threaten", "escalate": "war"}),
    }
    assert strategies == expected


def test_escalation_game_p1_count(escalation_game):
    """Player 1 has exactly 2 reduced strategies in the escalation game."""
    strategies = find_reduced_pure_strategies(escalation_game, player=1)
    assert len(strategies) == 2


def test_escalation_game_p1_strategies(escalation_game):
    """Player 1's reduced strategies are correct."""
    strategies = find_reduced_pure_strategies(escalation_game, player=1)
    expected = {
        Strategy({"threaten": "concede"}),
        Strategy({"threaten": "escalate"}),
    }
    assert strategies == expected


def test_escalation_game_no_empty_strategy_p0(escalation_game):
    """No spurious empty strategy is returned for player 0."""
    strategies = find_reduced_pure_strategies(escalation_game, player=0)
    assert Strategy({}) not in strategies


def test_escalation_game_no_empty_strategy_p1(escalation_game):
    """No spurious empty strategy is returned for player 1."""
    strategies = find_reduced_pure_strategies(escalation_game, player=1)
    assert Strategy({}) not in strategies


def test_escalation_game_no_partial_strategy_p0(escalation_game):
    """Partial strategy (root only, without escalate when threaten chosen)
    must not appear."""
    strategies = find_reduced_pure_strategies(escalation_game, player=0)
    assert Strategy({"root": "threaten"}) not in strategies


# ---------------------------------------------------------------------------
# Simple perfect information game (single decision node)
# ---------------------------------------------------------------------------


def test_simple_tree_p0_reduced_equals_full(simple_perfect_info_tree):
    """With a single decision node reduced == full strategies."""
    strategies = find_reduced_pure_strategies(simple_perfect_info_tree, player=0)
    expected = {
        Strategy({"root": "left"}),
        Strategy({"root": "right"}),
    }
    assert strategies == expected


def test_simple_tree_p1_no_decisions(simple_perfect_info_tree):
    """Player 1 has no decision nodes — returns one empty strategy."""
    strategies = find_reduced_pure_strategies(simple_perfect_info_tree, player=1)
    assert strategies == {Strategy({})}


# ---------------------------------------------------------------------------
# Perfect information game with two decisions per player
# ---------------------------------------------------------------------------


def test_two_decisions_p0_reduced(two_decisions_per_player_tree):
    """Player 0 has 2 reduced strategies (one decision node, two actions)."""
    strategies = find_reduced_pure_strategies(two_decisions_per_player_tree, player=0)
    expected = {
        Strategy({"root": "p1_l"}),
        Strategy({"root": "p1_r"}),
    }
    assert strategies == expected


def test_two_decisions_p1_reduced(two_decisions_per_player_tree):
    """Player 1 has 2 reduced strategies (singleton info sets, but both
    reachable regardless of P0's move, so all 4 combos remain and
    deduplicate to 4)."""
    strategies = find_reduced_pure_strategies(two_decisions_per_player_tree, player=1)
    # P1 has two independent decision nodes (p1_l, p1_r), each always
    # reachable (P0's choice doesn't cut either off for P1's perspective in
    # reduced strategies — both nodes are distinct info sets with distinct
    # actions).  So reduced == full: 2 x 2 = 4 strategies.
    assert len(strategies) == 4


# ---------------------------------------------------------------------------
# Imperfect information game — same actions at both nodes
# ---------------------------------------------------------------------------


def test_imperfect_info_p0_reduced(imperfect_info_same_actions_tree):
    """Player 0 has 2 reduced strategies (root: left or right)."""
    strategies = find_reduced_pure_strategies(
        imperfect_info_same_actions_tree, player=0
    )
    expected = {
        Strategy({"root": "p1_l"}),
        Strategy({"root": "p1_r"}),
    }
    assert strategies == expected


def test_imperfect_info_p1_reduced(imperfect_info_same_actions_tree):
    """Player 1 has 2 reduced strategies — one info set I1 with actions
    up / down (action labels from actions field)."""
    strategies = find_reduced_pure_strategies(
        imperfect_info_same_actions_tree, player=1
    )
    # Info set I1 is always reached (P0 always moves to a P1 node)
    # The actions are defined as ["up", "down"] in the info set
    expected = {
        Strategy({"I1": "up"}),
        Strategy({"I1": "down"}),
    }
    assert strategies == expected


def test_imperfect_info_p1_reduced_count(imperfect_info_same_actions_tree):
    """Player 1 has exactly 2 reduced strategies in the imperfect info game."""
    strategies = find_reduced_pure_strategies(
        imperfect_info_same_actions_tree, player=1
    )
    assert len(strategies) == 2


# ---------------------------------------------------------------------------
# Imperfect info with extra decision reachable only on some paths
# ---------------------------------------------------------------------------


def test_imperfect_info_with_extra_decision(imperfect_info_with_extra_decision_tree):
    """
    P1 has info set I1 (reached always), and info set I2 (reached only
    if P1 chooses 'down' at I1 and then P0 continues to I2).

    Reduced strategies should include I2 only when it's reachable.
    """
    strategies = find_reduced_pure_strategies(
        imperfect_info_with_extra_decision_tree, player=1
    )

    # P1 has 2 info sets: I1 always reached, I2 only reached if they choose "down" at I1
    # - I1="up" path: terminals directly, I2 never reached
    # - I1="down" path: goes to P0 continuation, then I2 is reached
    # So we expect: 1 strategy for I1="up" + 2 strategies for I1="down" (with I2=x or I2=y)
    # = 3 reduced strategies
    assert len(strategies) == 3
    assert Strategy({"I1": "up"}) in strategies
    assert Strategy({"I1": "down", "I2": "x"}) in strategies
    assert Strategy({"I1": "down", "I2": "y"}) in strategies


# ---------------------------------------------------------------------------
# Prisoner's Dilemma
# ---------------------------------------------------------------------------


def test_prisoners_dilemma_p0_reduced(prisoners_dilemma_tree):
    """P0 has 2 reduced strategies: cooperate or defect."""
    strategies = find_reduced_pure_strategies(prisoners_dilemma_tree, player=0)

    assert len(strategies) == 2
    decision_sets = [s.decisions for s in strategies]
    assert {"root": "cooperate"} in decision_sets
    assert {"root": "defect"} in decision_sets


def test_prisoners_dilemma_p1_reduced(prisoners_dilemma_tree):
    """P1 has 2 reduced strategies: cooperate or defect.

    P1's info set I1 is reached regardless of P0's choice, so both
    actions are part of all reduced strategies.
    """
    strategies = find_reduced_pure_strategies(prisoners_dilemma_tree, player=1)

    assert len(strategies) == 2
    decision_sets = [s.decisions for s in strategies]
    assert {"I1": "cooperate"} in decision_sets
    assert {"I1": "defect"} in decision_sets


# ---------------------------------------------------------------------------
# Empty game
# ---------------------------------------------------------------------------


def test_empty_game_returns_empty_strategy():
    """An empty game (no root) returns one empty strategy."""
    tree = GameTree(num_players=2)
    strategies = find_reduced_pure_strategies(tree, player=0)
    assert strategies == {Strategy({})}
