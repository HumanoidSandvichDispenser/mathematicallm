"""
Tests for strategy service - finding full pure strategies.
"""

import pytest
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    TerminalNodeData,
)
from zermelo.services.strategy_service import find_reduced_pure_strategies
from zermelo.extensive.strategy import Strategy


def test_escalation_game_returns_reduced_strategies(escalation_game_tree: GameTree):
    """Test that escalation game returns reduced pure strategies."""
    strategies = find_reduced_pure_strategies(escalation_game_tree, player=0)

    expected_p1_strategies = [
        Strategy({"p1_root", "accept"}),
        Strategy({"p1_root", "threaten"}, {"p1_again", "give_up"}),
        Strategy({"p1_root", "threaten"}, {"p1_again", "war"}),
    ]

    for expected in expected_p1_strategies:
        assert expected in strategies
