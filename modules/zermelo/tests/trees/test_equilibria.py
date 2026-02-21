"""
Tests for equilibria.py - finding Nash equilibria.
"""

import pytest
from sympy import S
from sympy.tensor.array.ndim_array import NDimArray
from zermelo.trees.strategy import Strategy
from zermelo.analysis.equilibria import find_pure_nash_equilibria


class TestFindPureNashEquilibria:
    def test_single_player_single_strategy(self):
        """Single player with single strategy is always a NE."""
        p0_strats = [Strategy({})]
        array = NDimArray([(S(5),)], (1, 1))

        ne = find_pure_nash_equilibria([p0_strats], array)

        assert len(ne) == 1
        assert ne[0] == (p0_strats[0],)

    def test_two_by_two_game_one_ne(self):
        """2x2 game with one pure NE at (0, 0)."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(3), S(3)), (S(0), S(5)), (S(5), S(0)), (S(1), S(1))],
            (2, 2, 2),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 1
        assert ne[0][0] == p0_strats[1]
        assert ne[0][1] == p1_strats[1]

    def test_two_by_two_game_two_ne(self):
        """2x2 coordination game with two pure NE at (0, 0) and (1, 1)."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(3), S(3)), (S(0), S(0)), (S(0), S(0)), (S(1), S(1))],
            (2, 2, 2),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 2
        assert (p0_strats[0], p1_strats[0]) in ne
        assert (p0_strats[1], p1_strats[1]) in ne

    def test_no_pure_ne(self):
        """2x2 matching pennies has no pure NE."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(1), S(-1)), (S(-1), S(1)), (S(-1), S(1)), (S(1), S(-1))],
            (2, 2, 2),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 0

    def test_three_player_game(self):
        """Three player game with one NE."""
        p0_strats = [Strategy({})]
        p1_strats = [Strategy({})]
        p2_strats = [Strategy({}), Strategy({})]

        array = NDimArray(
            [(S(3), S(3), S(3)), (S(1), S(1), S(1))],
            (1, 1, 2, 3),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats, p2_strats], array)

        assert len(ne) == 1

    def test_player_cannot_improve_unilaterally(self):
        """NE where each player cannot improve by deviating."""
        p0_strats = [Strategy({"root": "L"}), Strategy({"root": "R"})]
        p1_strats = [Strategy({"I1": "U"}), Strategy({"I1": "D"})]

        array = NDimArray(
            [(S(2), S(1)), (S(0), S(0)), (S(0), S(0)), (S(1), S(2))],
            (2, 2, 2),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 2
        assert (p0_strats[0], p1_strats[0]) in ne
        assert (p0_strats[1], p1_strats[1]) in ne
